#!/bin/sh
curl http://localhost:11434/api/generate -d '{
  "model": "gemma3:4b",
  "prompt": "Describe the element Hydrogen."
}'
