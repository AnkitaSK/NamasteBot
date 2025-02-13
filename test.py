# import langchain_google_community
# print(dir(langchain_google_community))


import asyncio
from rag_pipeline import rag_pipeline_with_translation

query = "Haridwar"
result = asyncio.run(rag_pipeline_with_translation(query))
print(result)
