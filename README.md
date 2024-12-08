# **Flask-ML-Deployment**

## **Objective**
This project aims to automate a distributed pipeline for AI applications by:
- Deploying AI models using Flask.
- Securing the application.
- Containerizing the application with Docker.
- Implementing multiple model endpoints (Object Recognition, Regression, and Text Generation).
- Automating deployment processes.

---

## **Implementation Details**

### **Step 1: Deploying a VGG16 Object Recognition Model**
- **Endpoint**: `/predict`
- **Description**: 
  - Accepts an image file.
  - Processes it using the pre-trained VGG16 model.
  - Returns the object label and confidence score.

### **Step 2: Securing File Uploads**
- Used `secure_filename` from `werkzeug.utils` to sanitize uploaded file names.
- Ensures only image files are accepted by validating file extensions.

### **Step 3: Containerizing the Application**
- **Dockerfile**:
    ```dockerfile
    FROM python:3.11-slim-buster
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    COPY . .
    EXPOSE 5000
    CMD [ "python", "app.py" ]
    ```
- **docker-compose.yml**:
    ```yaml
    version: '3.8'
    services:
      flask-app:
        build: .
        container_name: flask-app
        ports:
          - '5000:5000'
        environment:
          - FLASK_APP=app.py
          - FLASK_ENV=development
        volumes:
          - ./data:/app/data
    ```

### **Step 4: Adding a Regression Model Endpoint**
- **Endpoint**: `/regpredict`
- **Description**: 
  - Uses a pre-trained regression model (stored as a pickle file).
  - Predicts an employee's salary based on input features.

### **Step 5: Adding a Text Generation Endpoint**
- **Endpoint**: `/textgen`
- **Description**: 
  - Uses Hugging Face’s GPT pipeline to generate text based on a user-provided prompt.

### **Step 6: Adding a Home Route**
- **Endpoint**: `/home`
- **Description**: 
  - Serves as a documentation page explaining the project, its purpose, and deployment steps.

---

## **Deployment Instructions**

1. Clone the repository:
    ```bash
    git clone https://github.com/mohamed-stifi/Flask-ML-Deployment.git
    cd Flask-ML-Deployment
    cd app
    ```
2. Build the Docker container:
    ```bash
    docker-compose up --build
    ```
3. Wait until the container completes setup.
4. Access the application at `http://localhost:5000`.
5. Use the following routes:
   - `/predict`: Object recognition.
   - `/regpredict`: Regression predictions.
   - `/textgen`: Text generation.
   - `/home`: Project overview and documentation.

---

## **Evaluation**

- **Functionality**: All endpoints work as intended, demonstrating diverse AI capabilities.
- **Security**: File uploads sanitized using `secure_filename`.
- **Portability**: Application containerized for easy deployment across different environments.
- **Scalability**: Supports the addition of new models with minimal modifications.

---

## **Conclusion**

This project successfully demonstrates a distributed automation pipeline for AI applications using Flask, Docker, and modern AI frameworks. The modular design allows easy expansion for future AI model deployments.
