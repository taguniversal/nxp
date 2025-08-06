#!/bin/sh
curl -X POST http://localhost:4000/api/llm/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Describe the element Hydrogen."}'
