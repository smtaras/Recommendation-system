from fastapi import FastAPI
from pydantic import BaseModel, Field

from recommender import Recommender

app = FastAPI(title="Product Recommendation API")

recommender = Recommender()


class RecommendationRequest(BaseModel):
    visitor_id: int
    n: int = Field(default=10, ge=1, le=50)
    exclude_item_ids: list[int] = Field(default_factory=list)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommendations")
def get_recommendations(request: RecommendationRequest):
    items = recommender.recommend_items(
        visitor_id=request.visitor_id,
        n=request.n,
        exclude_item_ids=request.exclude_item_ids,
    )

    strategy = (
        "als"
        if request.visitor_id in recommender.user_to_index
        else "popular_fallback"
    )

    return {
        "visitor_id": request.visitor_id,
        "strategy": strategy,
        "recommendations": items,
    }