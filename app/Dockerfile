FROM pytorch/pytorch

# Step 4: Copy source code in the current directory to the container
ADD . /app

# Step 5: Set working directory to previously added app directory
WORKDIR /app

RUN pip --no-cache-dir install -r requirements.txt
# Step 6: Expose the port Flask is running on

# Step 9: Run Flask
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]