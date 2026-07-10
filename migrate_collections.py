import chromadb

def migrate():
    print("Initializing ChromaDB PersistentClient...")
    client = chromadb.PersistentClient(path='chroma_db')
    
    # 1. Check if source collection exists
    try:
        src_col = client.get_collection('siemens_manuals')
        print("Successfully accessed source collection 'siemens_manuals'.")
    except Exception as e:
        print("Source collection 'siemens_manuals' does not exist or cannot be accessed. Nothing to migrate.")
        return

    # 2. Get target collections (create them if they do not exist)
    print("Getting or creating destination collections...")
    col_1200 = client.get_or_create_collection('s7-1200')
    col_410 = client.get_or_create_collection('cpu-410')

    # 3. Migrate S7-1200 system manual
    s7_doc = 's71200_system_manual_en-US_en-US.pdf'
    print(f"Retrieving chunks for '{s7_doc}' from 'siemens_manuals'...")
    data_1200 = src_col.get(where={"source": s7_doc}, include=["embeddings", "metadatas", "documents"])
    
    if data_1200 and data_1200.get("ids"):
        ids = data_1200["ids"]
        embeddings = data_1200["embeddings"]
        metadatas = data_1200["metadatas"]
        documents = data_1200["documents"]
        print(f"Found {len(ids)} chunks for '{s7_doc}'. Copying to 's7-1200' collection...")
        col_1200.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        print(f"Successfully migrated '{s7_doc}' to 's7-1200'.")
    else:
        print(f"No chunks found for '{s7_doc}' in 'siemens_manuals'.")

    # 4. Migrate CPU 410 system manual
    cpu_doc = 'AS410-5H_System_Manuals7400_cpu_410_proc_autom_smart_system_system_en-US_en-US.pdf'
    print(f"Retrieving chunks for '{cpu_doc}' from 'siemens_manuals'...")
    data_410 = src_col.get(where={"source": cpu_doc}, include=["embeddings", "metadatas", "documents"])
    
    if data_410 and data_410.get("ids"):
        ids = data_410["ids"]
        embeddings = data_410["embeddings"]
        metadatas = data_410["metadatas"]
        documents = data_410["documents"]
        print(f"Found {len(ids)} chunks for '{cpu_doc}'. Copying to 'cpu-410' collection...")
        col_410.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        print(f"Successfully migrated '{cpu_doc}' to 'cpu-410'.")
    else:
        print(f"No chunks found for '{cpu_doc}' in 'siemens_manuals'.")

if __name__ == "__main__":
    migrate()
