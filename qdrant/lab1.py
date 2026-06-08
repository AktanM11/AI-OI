from qdrant_client import QdrantClient, models
import os
import dotenv

dotenv.load_dotenv()

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

collection_name = "test_collection"

# Create a collection
if client.collection_exists(collection_name):
    print(f"Collection '{collection_name}' already exists.")
else:
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "test": models.VectorParams(size=4, distance=models.Distance.COSINE)
        }
    )
    print(f"Collection '{collection_name}' created.")

# Insert some points
import random

points = []

for i in range(10):
    vector = [random.random() for _ in range(4)]
    points.append(models.PointStruct(id=i, vector={"test": vector}, payload={"name": f"point_{i}"}))

client.upsert(
    collection_name=collection_name,
    points=points
)

print("Inserted 10 points into the collection.")

# Try to insert a point with wrong vector size
# client.upsert(
#     collection_name=collection_name,
#     points=[
#         models.PointStruct(id=10, vector={"test": [0.1, 0.2]}, payload={"name": "invalid_point"})
#     ]
# )
# "status":{"error":"Wrong input: Vector dimension error: expected dim: 4, got 2 for vector \'test\'"}