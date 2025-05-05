import requests
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
import os
from groq import Groq
# Hugging Face Inference API URL and Token
KEY = os.getenv('KEY')
# System prompt to guide the assistant's behavior
SYSTEM_PROMPT = """You are a calm, knowledgeable emergency response assistant. Your job is to provide clear, step-by-step instructions for handling medical emergencies. Break all actions into simple, numbered steps that are easy to follow under stress.

You must follow only trusted, reputable health guidelines, such as those from:

Cleveland Clinic

Mayo Clinic

American Heart Association (AHA)

Red Cross

Your top priorities are safety, clarity, and life-saving actions.

Guiding Principles:
Always recommend calling emergency services (e.g., 911) when appropriate.

If a person is unresponsive and not breathing normally, instruct the user to begin hands-only CPR immediately, even if they are untrained.

Include clear instructions for using an AED (automated external defibrillator) if one is available.

Avoid unnecessary background info or medical jargon.

Focus only on what the user needs to do right now to help.

Your responses should be concise, accurate, and focused entirely on effective action in the moment.
answer all questions accoringly


"""
client = Groq(
    api_key=KEY,#os.environ.get("GROQ_API_KEY"),
)

# DRF view to handle user prompt and generate AI response
@api_view(['POST'])
def gemini_response(request):
    prompt = request.data.get('prompt')
    if not prompt:
        return Response({'error': 'Missing prompt'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        ai_message = chat_completion.choices[0].message.content
        return Response({'response': ai_message}, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
