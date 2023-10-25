## Run the Project:

Update the NGINX Configuration:

Assuming you have NGINX installed, you can update its configuration to redirect requests to the FastAPI microservice.

Start the FastAPI App and NGINX:

Start the FastAPI app on port 9192 using the following command:

    command: uvicorn main:app --host 0.0.0.0 --port 9192

This will make the FastAPI app accessible externally on port 9192.

Then, start NGINX and specify the path to the configuration file:

    nginx -c /path/to/fastapi.conf

Now, NGINX will redirect requests to the FastAPI microservice running on port 9192.
You can access the microservice through NGINX at http://localhost in your browser,
upload an image, and ask questions using the form. The NGINX server will proxy the requests to the FastAPI app for
processing.