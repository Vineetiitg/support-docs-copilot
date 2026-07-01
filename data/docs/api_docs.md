# API Documentation

## Rate Limits
The default API rate limit is 100 requests per minute per IP address. If you exceed this limit, you will receive an HTTP 429 Too Many Requests response.

## Authentication
Authentication is performed via JWT tokens. Include the token in the `Authorization` header as a Bearer token:
`Authorization: Bearer <token>`

## Error Codes
- **401 Unauthorized**: The token is missing or invalid.
- **403 Forbidden**: You do not have permission to access the resource.
- **404 Not Found**: The requested resource could not be found. Check your router configuration.
- **429 Too Many Requests**: You have exceeded the rate limit.
- **500 Internal Server Error**: An unexpected error occurred on the server.
