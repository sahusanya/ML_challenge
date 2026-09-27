import sqlite3
import re
from pathlib import Path
from collections import Counter

import pandas as pd

from .preprocess import normalize_text


class ScalableBlocker:

    def __init__(self, db_path="models/blocker_test.db"):
        self.db_path = db_path

    # =========================================================
    # TOKENIZATION
    # =========================================================

    @staticmethod
    def _tokens(text):
        if not text:
            return []

        return [
            token
            for token in re.findall(
                r"[^\W_]+",
                str(text).lower(),
                flags=re.UNICODE
            )
            if len(token) >= 2
        ]

    @staticmethod
    def _address_tokens(text):
        """
        Address-specific tokenization.

        Keeps:
        - words
        - numbers
        - alphanumeric identifiers

        Examples:
            D-4       -> d, 4
            187C      -> 187c
            113/154   -> 113, 154
        """

        if not text:
            return []

        text = str(text).lower()

        tokens = re.findall(
            r"[a-z0-9]+",
            text
        )

        return [
            token
            for token in tokens
            if len(token) >= 2 or token.isdigit()
        ]

    @staticmethod
    def _fts_escape(token):
        return token.replace('"', '""')

    # =========================================================
    # INDEX CREATION
    # =========================================================

    def create_index(
        self,
        source2_path,
        source3_path,
        chunksize=50_000
    ):

        db_path = Path(self.db_path)
        db_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if db_path.exists():
            db_path.unlink()

        conn = sqlite3.connect(
            str(db_path)
        )

        # -----------------------------------------------------
        # FTS INDEX
        # -----------------------------------------------------

        conn.execute("""
            CREATE VIRTUAL TABLE candidates
            USING fts5(
                entity_id UNINDEXED,
                source UNINDEXED,
                country,
                business_name,
                business_address,
                prefix='3'
            )
        """)

        # -----------------------------------------------------
        # ADDRESS TOKEN INDEX
        # -----------------------------------------------------

        conn.execute("""
            CREATE TABLE address_tokens (
                token TEXT,
                entity_id TEXT,
                source TEXT,
                country TEXT
            )
        """)

        conn.execute("""
            CREATE INDEX idx_address_token
            ON address_tokens(token, country)
        """)

        conn.execute(
            "PRAGMA journal_mode=WAL"
        )

        conn.execute(
            "PRAGMA synchronous=OFF"
        )

        total = 0

        # =====================================================
        # READ S2 + S3
        # =====================================================

        for source, path in [
            ("S2", source2_path),
            ("S3", source3_path)
        ]:

            print(
                f"\nIndexing {source}: {path}"
            )

            for chunk in pd.read_csv(
                path,
                sep="\t",
                dtype=str,
                chunksize=chunksize
            ):

                chunk = chunk.fillna("")

                chunk["norm_name"] = (
                    chunk["business_name"]
                    .apply(normalize_text)
                )

                chunk["norm_address"] = (
                    chunk["business_address"]
                    .apply(normalize_text)
                )

                # -------------------------------------------------
                # FTS rows
                # -------------------------------------------------

                rows = []

                for row in chunk.itertuples():

                    rows.append(
                        (
                            row.entity_id,
                            source,
                            row.country,
                            row.norm_name,
                            row.norm_address
                        )
                    )

                conn.executemany(
                    """
                    INSERT INTO candidates
                    (
                        entity_id,
                        source,
                        country,
                        business_name,
                        business_address
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    rows
                )

                # -------------------------------------------------
                # Address token rows
                # -------------------------------------------------

                address_rows = []

                for row in chunk.itertuples():

                    tokens = set(
                        ScalableBlocker
                        ._address_tokens(
                            row.norm_address
                        )
                    )

                    for token in tokens:

                        address_rows.append(
                            (
                                token,
                                row.entity_id,
                                source,
                                row.country
                            )
                        )

                conn.executemany(
                    """
                    INSERT INTO address_tokens
                    (
                        token,
                        entity_id,
                        source,
                        country
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    address_rows
                )

                conn.commit()

                total += len(chunk)

                print(
                    f"\rIndexed {total:,} candidate records",
                    end=""
                )

        conn.close()

        print(
            f"\n\nIndex created: {db_path}"
        )

        print(
            f"Total indexed records: {total:,}"
        )

    # =========================================================
    # FTS SEARCH
    # =========================================================

    def _search(
        self,
        conn,
        country,
        field,
        tokens,
        limit=50
    ):
        if not tokens:
            return []

        tokens = sorted(
            set(tokens),
            key=len,
            reverse=True
        )[:8]

        conditions = []

        for token in tokens:
            token = self._fts_escape(token)

            conditions.append(
                f'{field}:"{token}"*'
            )

        query = " OR ".join(conditions)

        sql = f"""
            SELECT
                entity_id,
                source,
                country,
                business_name,
                business_address
            FROM candidates
            WHERE candidates MATCH ?
            AND country = ?
            ORDER BY bm25(candidates)
            LIMIT ?
        """

        return conn.execute(
            sql,
            (
                query,
                country,
                limit
            )
        ).fetchall()

    # =========================================================
    # ADDRESS ANCHOR SEARCH
    # =========================================================

    def _address_anchor_search(
        self,
        conn,
        country,
        tokens,
        limit=100
    ):
        if not tokens:
            return set()

        useful_tokens = []

        for token in set(tokens):
            if token.isdigit() or len(token) >= 4:
                useful_tokens.append(token)

        useful_tokens = sorted(
            useful_tokens,
            key=len,
            reverse=True
        )[:12]

        if not useful_tokens:
            return set()

        placeholders = ",".join(["?"] * len(useful_tokens))

        query = f"""
            SELECT
                entity_id,
                source,
                COUNT(*) AS overlap
            FROM address_tokens
            WHERE country = ?
            AND token IN ({placeholders})
            GROUP BY entity_id, source
            ORDER BY overlap DESC
            LIMIT ?
        """

        params = [country] + useful_tokens + [limit]

        rows = conn.execute(
            query,
            params
        ).fetchall()

        return {
            (entity_id, source)
            for entity_id, source, overlap in rows
        }

    # =========================================================
    # MAIN CANDIDATE GENERATION
    # =========================================================
    def get_candidate_records(
        self,
        business_name,
        business_address,
        country,
        name_limit=50,
        address_limit=50,
        conn=None
    ):
        own_connection = False

        if conn is None:
            conn = sqlite3.connect(self.db_path)
            own_connection = True

        norm_name = normalize_text(business_name)
        norm_address = normalize_text(business_address)

        name_tokens = self._tokens(norm_name)
        address_tokens = self._address_tokens(norm_address)

        candidates = {}

        # Name FTS
        for row in self._search(
            conn,
            country,
            "business_name",
            name_tokens,
            name_limit
        ):
            entity_id, source, row_country, name, address = row
            candidates[(entity_id, source)] = row

        # Address FTS
        for row in self._search(
            conn,
            country,
            "business_address",
            address_tokens,
            address_limit
        ):
            entity_id, source, row_country, name, address = row
            candidates[(entity_id, source)] = row

        if own_connection:
            conn.close()

        return list(candidates.values())

    def get_candidates(
        self,
        business_name,
        business_address,
        country,
        name_limit=50,
        address_limit=50,
        conn=None
    ):
        own_connection = False

        if conn is None:
            conn = sqlite3.connect(self.db_path)
            own_connection = True

        norm_name = normalize_text(business_name)
        norm_address = normalize_text(business_address)

        name_tokens = self._tokens(norm_name)
        address_tokens = self._address_tokens(norm_address)

        candidates = set()

        # -----------------------------------------------------
        # 1. Name FTS
        # -----------------------------------------------------

        candidates.update(
            self._search(
                conn,
                country,
                "business_name",
                name_tokens,
                name_limit
            )
        )

        # -----------------------------------------------------
        # 2. Address FTS
        # -----------------------------------------------------

        candidates.update(
            self._search(
                conn,
                country,
                "business_address",
                address_tokens,
                address_limit
            )
        )

        # -----------------------------------------------------
        # 3. Address anchor index
        # -----------------------------------------------------

        candidates.update(
            self._address_anchor_search(
                conn,
                country,
                address_tokens,
                limit=100
            )
        )

        if own_connection:
            conn.close()

        return list(candidates)