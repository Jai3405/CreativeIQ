from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.services.design_coach import DesignCoach

router = APIRouter()

# Global coach instance
coach = DesignCoach()


@router.post("/", response_model=ChatResponse)
async def chat_with_design_coach(request: ChatRequest):
    """
    Conversational interface for design feedback and questions
    """
    try:
        response = await coach.process_message(request)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.post("/feedback")
async def submit_feedback(
    analysis_id: str,
    rating: int,
    comments: str = ""
):
    """
    Submit feedback on analysis quality
    """
    if not 1 <= rating <= 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")

    try:
        await coach.process_feedback(analysis_id, rating, comments)
        return {"message": "Feedback submitted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")