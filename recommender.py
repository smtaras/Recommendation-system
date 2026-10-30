from pathlib import Path
import pickle

from scipy.sparse import load_npz
from implicit.cpu.als import AlternatingLeastSquares


class Recommender:
    def __init__(self, artifacts_dir=None):
        if artifacts_dir is None:
            artifacts_dir = Path(__file__).resolve().parent / "artifacts"

        artifacts_dir = Path(artifacts_dir)

        self.model = AlternatingLeastSquares.load(
            str(artifacts_dir / "als_model.npz")
        )
        self.user_item_matrix = load_npz(
            artifacts_dir / "user_item_matrix.npz"
        )

        with open(artifacts_dir / "metadata.pkl", "rb") as file:
            metadata = pickle.load(file)

        self.user_to_index = metadata["user_to_index"]
        self.item_ids = metadata["item_ids"]
        self.popular_item_ids = metadata["popular_item_ids"]

    def recommend_items(self, visitor_id, n=10, exclude_item_ids=None):
        if n < 1:
            raise ValueError("n must be at least 1")

        visitor_id = int(visitor_id)
        excluded_items = {
            int(item_id) for item_id in (exclude_item_ids or [])
        }

        recommendations = []
        user_index = self.user_to_index.get(visitor_id)

        if user_index is not None:
            candidate_count = min(
                len(self.item_ids),
                n + len(excluded_items),
            )

            item_indices, _ = self.model.recommend(
                user_index,
                self.user_item_matrix[user_index],
                N=candidate_count,
                filter_already_liked_items=True,
            )

            recommendations = [
                int(self.item_ids[index])
                for index in item_indices
                if int(self.item_ids[index]) not in excluded_items
            ]

            seen_indices = self.user_item_matrix[user_index].indices
            seen_items = {
                int(self.item_ids[index]) for index in seen_indices
            }
        else:
            seen_items = set()

        blocked_items = excluded_items | seen_items

        for item_id in self.popular_item_ids:
            item_id = int(item_id)

            if (
                item_id not in blocked_items
                and item_id not in recommendations
            ):
                recommendations.append(item_id)

            if len(recommendations) >= n:
                break

        return recommendations[:n]