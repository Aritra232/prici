# Pricila - AI-Powered Hobby & Industry Search API

This is a FastAPI-based web application that uses OpenAI to provide intelligent search functionalities for hobbies, industries, and job titles. It handles typos and partial inputs automatically.

## Requirements

- Python 3.8+
- OpenAI API Key

## Installation

1. Navigate to the project directory.
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows, use:
   venv\Scripts\activate
   # On macOS/Linux, use:
   # source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your environment variables. Ensure you have your `OPENAI_API_KEY` set up (either exported or stored in a `.env` file):
   ```env
   OPENAI_API_KEY=sk-...
   ```

## Running the Application

You can start the FastAPI application using `uvicorn`. Run the following command in your terminal from the root of the project:

```bash
uvicorn main:app --reload
```

This will start a local development server. 
- `main:app` refers to the `app` instance in `main.py`.
- The `--reload` flag enables auto-reloading so the server will restart automatically when you make changes to the code (useful for development).
- By default, the API will be available at: `http://127.0.0.1:8000`

Alternatively, you can also run the project directly using Python, as `uvicorn.run()` is configured in the main block:
```bash
python main.py
```

### API Documentation

Once the server is running, you can access the interactive API documentation at:
- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## Endpoints

- `GET /health` - Health check
- `GET /hobbies/search?q={query}&limit={limit}` - Search for hobbies
- `GET /industries/search?q={query}&limit={limit}` - Search for industries
- `GET /job-titles/search?q={query}&limit={limit}` - Search for job titles
