import os
from time import sleep
from dotenv import load_dotenv
from google import genai
import json

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

def rerank_individual(results, query, documents, limit):
    
    for result in results:
        doc = documents[int(result["id"]) - 1]
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("description", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""
    
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        result["rerank_score"] = float(response.text)
        sleep(3)

    results.sort(key=lambda x: x["rerank_score"], reverse=True)

    results = results[:limit]
    
    return results
    
def rerank_batch(results, query, documents, limit):

    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{results}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    new_ranking = json.loads(response.text)
 
    sorted_results = sorted(results, key=lambda x: new_ranking.index(int(x["id"])), reverse=True)

    for i, result in enumerate(sorted_results):
        result["rerank_score"] = i + 1
    
    results = sorted_results[:limit]
    
    return results

def rerank(results, rerank_method, query, documents, limit):
    if rerank_method == "individual":
        return rerank_individual(results, query, documents, limit)
    elif rerank_method == "batch":
        return rerank_batch(results, query, documents, limit)