# RAG Evaluation Report

Generated: 2026-07-04T11:31:04.338800+00:00

## Overall Metrics
- **Questions Evaluated:** 5
- **Source Hit Rate:** 0.6
- **Average Latency:** 13929.41 ms

### Ragas Scores
```json
{
  "error": "'ChatOpenAI' object has no attribute 'set_run_config'"
}
```

## Question Results

### Q: What is Error Code 404?
**A:** Error Code 404 indicates that the requested server resource was not found, and you should check your network router configuration [sample_error.txt][api_docs.md].
- Source hit: True
- Retrieved contexts: 3
- Latency ms: 39687.2

### Q: What does this support copilot do?
**A:** This support copilot is designed to answer support documentation queries using retrieved context, currently handling each query independently without conversation memory [implementation_plan.md].
- Source hit: False
- Retrieved contexts: 3
- Latency ms: 12167.86

### Q: How do I reset my password?
**A:** 
- Source hit: False
- Retrieved contexts: 0
- Latency ms: 5999.2

### Q: What is the default rate limit?
**A:** The default API rate limit is 100 requests per minute per IP address. [api_docs.md]
- Source hit: True
- Retrieved contexts: 3
- Latency ms: 5868.04

### Q: How can I contact support?
**A:** You can contact support via email at support@example.com or by phone at 1-800-555-0199, available Monday–Friday, 9 AM – 5 PM EST. [contact_info.html] For urgent issues, call the phone number above and press 1 for priority routing. [contact_info.html]
- Source hit: True
- Retrieved contexts: 3
- Latency ms: 5924.76
