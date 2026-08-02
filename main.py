import os
import re
import httpx
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="NexusLink AI Engine")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
VERCEL_TOKEN = os.getenv("VERCEL_TOKEN")

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Prompt describing the app to generate.")
    project_slug: str = Field(..., description="Unique URL slug prefix.")

class GenerateResponse(BaseModel):
    status: str
    live_url: str
    deployment_id: str

def sanitize_html(raw_code: str) -> str:
    cleaned = re.sub(r"^```html\s*", "", raw_code, flags=re.MULTILINE)
    cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()

async def deploy_to_vercel(html_content: str, slug: str) -> dict:
    if not VERCEL_TOKEN:
        raise HTTPException(status_code=500, detail="VERCEL_TOKEN missing in .env")

    headers = {
        "Authorization": f"Bearer {VERCEL_TOKEN}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "name": slug,
        "files": [{"file": "index.html", "data": html_content}],
        "projectSettings": {"framework": None}
    }

    async with httpx.AsyncClient() as http_client:
        response = await http_client.post("https://api.vercel.com/v13/deployments", json=payload, headers=headers)
        if response.status_code not in (200, 201):
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
        data = response.json()
        return {"live_url": f"https://{data['url']}", "deployment_id": data.get("id", "")}

# Serve the visual UI directly
@app.get("/", response_class=HTMLResponse)
async def home():
    index_path = Path(__file__).parent / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html file was not found in project directory.")
    return index_path.read_text(encoding="utf-8")

# API Endpoint for generation
@app.post("/api/v1/generate", response_model=GenerateResponse)
async def generate_and_deploy(request: GenerateRequest):
    system_prompt = (
        "You are an expert full-stack engineer producing standalone micro-applications. "
        "Generate a complete, single-file index.html document containing inline Tailwind CSS (via CDN) "
        "and clean JavaScript for interactive elements. "
        "IMPORTANT: Output ONLY valid raw HTML code. Do NOT include markdown blocks."
    )
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.prompt}
            ],
            temperature=0.7,
        )
        
        raw_code = completion.choices[0].message.content
        cleaned_html = sanitize_html(raw_code)
        deployment_data = await deploy_to_vercel(cleaned_html, request.project_slug)
        
        return GenerateResponse(
            status="published",
            live_url=deployment_data["live_url"],
            deployment_id=deployment_data["deployment_id"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))