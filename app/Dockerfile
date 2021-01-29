FROM pytorch/pytorch

# Step 1: Copy source code in the current directory to the container
ADD . /app

# Step 2: Set working directory to previously added app directory
WORKDIR /app

# Step 3: Install requirements
RUN pip --no-cache-dir install -r requirements.txt

# Step 4: Run Flask
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]