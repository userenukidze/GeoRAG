import fs from "fs";
import { pipeline } from "@xenova/transformers";
import { Pinecone } from "@pinecone-database/pinecone";

// ===== CONFIGURATION CONSTANTS =====

// File Processing Configuration
const SOURCE_TEXT_FILE_PATH = "./Dummy Text Files/corpora.txt";
const WORDS_PER_CHUNK = 500;
const OVERLAP_WORDS = 50;
const BATCH_SIZE = 100;

// Pinecone Configuration
// SECURITY NOTE: In production, use environment variables
const PINECONE_API_KEY = "pcsk_TqZYd_2JKFQdA9hNpduqVHPx2E6Xo5LfQsLNZFRjXMDu4jnWWrtdpNkitXNs96cTHWUec";
const PINECONE_INDEX_NAME = "my-bge-m3-index";
const PINECONE_INDEX_REGION = "us-east-1";

// Model Configuration
const EMBEDDING_MODEL_NAME = "Xenova/bge-m3";
const EMBEDDING_DIMENSION = 1024;
const SIMILARITY_METRIC = "cosine";

// Initialize Pinecone client
const pineconeClient = new Pinecone({
  apiKey: PINECONE_API_KEY,
});

// ===== TEXT PROCESSING FUNCTIONS =====

/**
 * Loads text from a file and performs basic cleaning
 * @param {string} filePath - Path to the text file
 * @returns {string} Cleaned text content
 */
function loadAndCleanTextFile(filePath) {
  try {
    console.log(`📖 Loading text from: ${filePath}`);
    const rawTextContent = fs.readFileSync(filePath, "utf-8");
    const cleanedText = rawTextContent.replace(/\s+/g, " ").trim();
    
    console.log(`✅ Successfully loaded ${cleanedText.split(' ').length} words`);
    return cleanedText;
    
  } catch (error) {
    console.error(`❌ Error loading text file ${filePath}:`, error.message);
    throw new Error(`Failed to load text file: ${error.message}`);
  }
}

/**
 * Splits text into overlapping chunks for better context preservation
 * @param {string} textContent - Input text to chunk
 * @param {number} wordsPerChunk - Maximum words per chunk
 * @param {number} overlapWords - Words to overlap between chunks
 * @returns {Array<Object>} Array of chunk objects with id and text
 */
function createTextChunks(textContent, wordsPerChunk = WORDS_PER_CHUNK, overlapWords = OVERLAP_WORDS) {
  console.log(`✂️ Creating text chunks (${wordsPerChunk} words per chunk, ${overlapWords} overlap)...`);
  
  const wordsArray = textContent.split(" ");
  const textChunks = [];
  
  for (let startIndex = 0; startIndex < wordsArray.length; startIndex += wordsPerChunk - overlapWords) {
    const chunkWords = wordsArray.slice(startIndex, startIndex + wordsPerChunk);
    
    textChunks.push({
      id: `chunk_${textChunks.length}`,
      text: chunkWords.join(" "),
      wordCount: chunkWords.length,
      startPosition: startIndex,
    });
  }
  
  console.log(`✅ Created ${textChunks.length} text chunks`);
  return textChunks;
}

// ===== EMBEDDING FUNCTIONS =====

/**
 * Generates embeddings for text chunks using the BGE-M3 model
 * @param {Array<Object>} textChunks - Array of text chunk objects
 * @returns {Promise<Array<Float32Array>>} Array of embedding vectors
 */
async function generateEmbeddingsForChunks(textChunks) {
  console.log(`🧠 Initializing embedding model '${EMBEDDING_MODEL_NAME}'...`);
  
  const embeddingPipeline = await pipeline("feature-extraction", EMBEDDING_MODEL_NAME);
  console.log("✅ Embedding model loaded successfully");

  const embeddingVectors = [];
  const totalChunks = textChunks.length;
  
  for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
    // Progress logging
    if (chunkIndex % 10 === 0 || chunkIndex === totalChunks - 1) {
      console.log(`⚡ Processing chunk ${chunkIndex + 1} of ${totalChunks}...`);
    }
    
    const embeddingOutput = await embeddingPipeline(textChunks[chunkIndex].text, {
      pooling: "cls",
      normalize: true,
    });
    
    embeddingVectors.push(embeddingOutput.data);
  }
  
  console.log(`✅ Generated ${embeddingVectors.length} embedding vectors`);
  return embeddingVectors;
}

/**
 * Generates embedding for a single query string
 * @param {string} queryText - Text to embed
 * @returns {Promise<Float32Array>} Query embedding vector
 */
async function generateQueryEmbedding(queryText) {
  console.log(`🧠 Embedding query: "${queryText}"...`);
  
  const embeddingPipeline = await pipeline("feature-extraction", EMBEDDING_MODEL_NAME);
  const queryEmbeddingOutput = await embeddingPipeline(queryText, {
    pooling: "cls",
    normalize: true,
  });
  
  console.log("✅ Query embedding generated");
  return queryEmbeddingOutput.data;
}

// ===== PINECONE DATABASE FUNCTIONS =====

/**
 * Ensures the Pinecone index exists, creates it if necessary
 * @returns {Promise<void>}
 */
async function ensurePineconeIndexExists() {
  console.log("🔍 Checking if Pinecone index exists...");
  
  const indexList = await pineconeClient.listIndexes();
  const indexExists = indexList.indexes?.some(index => index.name === PINECONE_INDEX_NAME);
  
  if (!indexExists) {
    console.log(`📝 Creating Pinecone index '${PINECONE_INDEX_NAME}'...`);
    
    await pineconeClient.createIndex({
      name: PINECONE_INDEX_NAME,
      dimension: EMBEDDING_DIMENSION,
      metric: SIMILARITY_METRIC,
      spec: {
        serverless: {
          cloud: "aws",
          region: PINECONE_INDEX_REGION,
        },
      },
    });
    
    console.log(`✅ Pinecone index '${PINECONE_INDEX_NAME}' created successfully`);
  } else {
    console.log(`✅ Pinecone index '${PINECONE_INDEX_NAME}' already exists`);
  }
}

/**
 * Uploads text chunks and embeddings to Pinecone in batches
 * @param {Array<Object>} textChunks - Text chunk objects
 * @param {Array<Float32Array>} embeddingVectors - Corresponding embeddings
 * @returns {Promise<void>}
 */
async function uploadChunksToPinecone(textChunks, embeddingVectors) {
  console.log(`💾 Preparing to upload ${textChunks.length} vectors to Pinecone...`);
  
  const pineconeIndex = pineconeClient.Index(PINECONE_INDEX_NAME);
  const vectorsToUpload = [];

  // Prepare vectors for upload
  for (let chunkIndex = 0; chunkIndex < textChunks.length; chunkIndex++) {
    const currentChunk = textChunks[chunkIndex];
    const currentEmbedding = embeddingVectors[chunkIndex];

    vectorsToUpload.push({
      id: currentChunk.id,
      values: Array.from(currentEmbedding),
      metadata: {
        text: currentChunk.text,
        wordCount: currentChunk.wordCount,
        startPosition: currentChunk.startPosition,
        chunkId: currentChunk.id,
      },
    });
  }

  // Upload in batches for efficiency
  const totalBatches = Math.ceil(vectorsToUpload.length / BATCH_SIZE);
  console.log(`📦 Uploading in ${totalBatches} batches of ${BATCH_SIZE}...`);
  
  for (let batchIndex = 0; batchIndex < vectorsToUpload.length; batchIndex += BATCH_SIZE) {
    const currentBatch = vectorsToUpload.slice(batchIndex, batchIndex + BATCH_SIZE);
    const batchNumber = Math.floor(batchIndex / BATCH_SIZE) + 1;
    
    try {
      await pineconeIndex.upsert(currentBatch);
      console.log(`✅ Uploaded batch ${batchNumber} of ${totalBatches}`);
      
    } catch (error) {
      console.error(`❌ Failed to upload batch ${batchNumber}:`, error.message);
      throw new Error(`Batch upload failed: ${error.message}`);
    }
  }

  console.log(`🎉 Successfully uploaded all ${vectorsToUpload.length} vectors to Pinecone`);
}

/**
 * Searches the Pinecone index for similar text chunks
 * @param {string} searchQuery - Query string to search for
 * @param {number} maxResults - Number of top results to return
 * @returns {Promise<void>}
 */
async function searchSimilarChunks(searchQuery, maxResults = 5) {
  console.log(`🔎 Searching for: "${searchQuery}"`);
  
  const pineconeIndex = pineconeClient.Index(PINECONE_INDEX_NAME);
  
  try {
    // Generate embedding for the search query
    const queryEmbedding = await generateQueryEmbedding(searchQuery);
    
    // Search the index
    console.log(`🔍 Querying Pinecone index for ${maxResults} similar chunks...`);
    const searchResults = await pineconeIndex.query({
      vector: Array.from(queryEmbedding),
      topK: maxResults,
      includeMetadata: true,
    });

    // Display results
    if (searchResults.matches && searchResults.matches.length > 0) {
      console.log(`✨ Found ${searchResults.matches.length} relevant results:\n`);
      
      searchResults.matches.forEach((match, rankIndex) => {
        const similarityScore = (match.score * 100).toFixed(2);
        const previewText = match.metadata?.text?.slice(0, 300) || "No text available";
        const truncationIndicator = match.metadata?.text?.length > 300 ? "..." : "";
        
        console.log(`🔹 Rank ${rankIndex + 1} | Similarity: ${similarityScore}%`);
        console.log(`📄 Chunk ID: ${match.id}`);
        console.log(`📝 Preview: ${previewText}${truncationIndicator}`);
        console.log("─".repeat(80) + "\n");
      });
    } else {
      console.log("❌ No relevant chunks found for your search query");
    }
    
  } catch (error) {
    console.error("❌ Search failed:", error.message);
    throw new Error(`Search operation failed: ${error.message}`);
  }
}

// ===== MAIN PIPELINE =====

/**
 * Main execution pipeline that orchestrates the entire process
 * @returns {Promise<void>}
 */
async function executeMainPipeline() {
  console.log("🚀 Starting Text Embedding and Search Pipeline\n");
  
  try {
    // Step 1: Initialize Pinecone
    console.log("=".repeat(50));
    console.log("STEP 1: PINECONE INITIALIZATION");
    console.log("=".repeat(50));
    await ensurePineconeIndexExists();

    // Step 2: Load and process text
    console.log("\n" + "=".repeat(50));
    console.log("STEP 2: TEXT PROCESSING");
    console.log("=".repeat(50));
    const cleanedTextContent = loadAndCleanTextFile(SOURCE_TEXT_FILE_PATH);
    const textChunks = createTextChunks(cleanedTextContent, WORDS_PER_CHUNK, OVERLAP_WORDS);

    // Step 3: Generate embeddings
    console.log("\n" + "=".repeat(50));
    console.log("STEP 3: EMBEDDING GENERATION");
    console.log("=".repeat(50));
    const embeddingVectors = await generateEmbeddingsForChunks(textChunks);

    // Step 4: Upload to Pinecone
    console.log("\n" + "=".repeat(50));
    console.log("STEP 4: PINECONE UPLOAD");
    console.log("=".repeat(50));
    await uploadChunksToPinecone(textChunks, embeddingVectors);

    // Step 5: Test search functionality
    console.log("\n" + "=".repeat(50));
    console.log("STEP 5: SEARCH TESTING");
    console.log("=".repeat(50));
    await searchSimilarChunks("What is the main topic discussed?", 3);
    
    console.log("\n🎉 Pipeline completed successfully!");
    
  } catch (error) {
    console.error("\n❌ Pipeline failed:", error.message);
    console.error("Stack trace:", error.stack);
    process.exit(1);
  }
}

// ===== EXECUTION =====

// Execute the main pipeline with comprehensive error handling
executeMainPipeline().catch((error) => {
  console.error("💥 Unhandled error in main pipeline:", error.message);
  console.error("Full error:", error);
  process.exit(1);
});