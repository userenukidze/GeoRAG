import fs from "fs";
import { pipeline, AutoTokenizer } from "@xenova/transformers";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";
import { env } from "process";
import dotenv from "dotenv";
dotenv.config();
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
// ===== CONFIGURATION CONSTANTS =====

// File Processing Configuration
const SOURCE_TEXT_FILE_PATH = "./Dummy Text Files/corpora.txt";
const BATCH_SIZE = 100;

// Pinecone Configuration
const PINECONE_API_KEY = process.env.PINECONE_API_KEY ;
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

function splitIntoSentences(text) {
  return text.match(/[^\.!\?]+[\.!\?]+/g) || [text];
}

async function createTextChunks(
  textContent,
  chunkSize = 500  // Fixed character size instead of token count
) {
  console.log(`✂️ Creating fixed-size chunks (${chunkSize} characters each)...`);
  
  const textChunks = [];
  let position = 0;
  
  // Split text into fixed-size chunks
  while (position < textContent.length) {
    // Extract the chunk
    const chunkText = textContent.slice(position, position + chunkSize);
    
    // Create the chunk object
    textChunks.push({
      id: `chunk_${textChunks.length}`,
      text: chunkText,
      wordCount: chunkText.split(' ').length,
      startPosition: position,
      charCount: chunkText.length
    });
    
    // Move to the next chunk position
    position += chunkSize;
  }
  
  console.log(`✅ Created ${textChunks.length} text chunks of ${chunkSize} characters each`);
  return textChunks;
}

// ===== EMBEDDING FUNCTIONS =====

async function generateEmbeddingsForChunks(textChunks) {
  console.log(`🧠 Initializing embedding model '${EMBEDDING_MODEL_NAME}'...`);
  const embeddingPipeline = await pipeline("feature-extraction", EMBEDDING_MODEL_NAME);
  console.log("✅ Embedding model loaded successfully");

  const embeddingVectors = [];
  const totalChunks = textChunks.length;

  for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
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

async function uploadChunksToPinecone(textChunks, embeddingVectors) {
  console.log(`💾 Preparing to upload ${textChunks.length} vectors to Pinecone...`);
  const pineconeIndex = pineconeClient.Index(PINECONE_INDEX_NAME);
  const vectorsToUpload = [];

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

// ===== AI-POWERED ANSWERING FUNCTION =====

async function answerWithGPT4O(question, topChunks) {
  const context = topChunks.map((c, i) => `Context ${i + 1}: ${c.text}`).join('\n\n');
  const prompt = `
You are an expert assistant. Use ONLY the following context to answer the user's question.

${context}

Question: ${question}
Answer:
  `;
  const completion = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: prompt }],
    max_tokens: 512,
    temperature: 0.2,
  });
  return completion.choices[0].message.content.trim();
}

/**
 * Searches the Pinecone index for similar text chunks and answers the question using GPT-4o
 * @param {string} searchQuery - Query string to search for
 * @param {number} maxResults - Number of top results to return
 * @param {Object} options - Additional options for the search
 * @returns {Promise<Object>} - Contains answer and source information
 */

async function searchAndAnswer(searchQuery, maxResults = 5, options = {}) {
  const { requestInfo = null } = options;

  let textFilePath = null;

  // If a file is uploaded, use its path instead of the default corpus
  if (requestInfo && requestInfo.files && requestInfo.files.length > 0) {
    // Multer uses .path for file path
    textFilePath = requestInfo.files[0].path || requestInfo.files[0].filepath;
    console.log(`📂 Using uploaded file: ${textFilePath}`);
  }

  // Load and process the selected file
  const cleanedTextContent = loadAndCleanTextFile(textFilePath);
  const textChunks = await createTextChunks(cleanedTextContent, 500);
  const embeddingVectors = await generateEmbeddingsForChunks(textChunks);
  await uploadChunksToPinecone(textChunks, embeddingVectors);

  // Now proceed with the search as before
  console.log(`🔎 Searching for: "${searchQuery}"`);
  const pineconeIndex = pineconeClient.Index(PINECONE_INDEX_NAME);

  try {
    const queryEmbedding = await generateQueryEmbedding(searchQuery);
    console.log(`🔍 Querying Pinecone index for ${maxResults} similar chunks...`);
    const searchResults = await pineconeIndex.query({
      vector: Array.from(queryEmbedding),
      topK: maxResults,
      includeMetadata: true,
    });

    let answer = null;
    let sources = [];

    if (searchResults.matches && searchResults.matches.length > 0) {
      console.log(`✨ Found ${searchResults.matches.length} relevant results:\n`);
      sources = searchResults.matches.map((match, rankIndex) => {
        const similarityScore = (match.score * 100).toFixed(2);
        const previewText = match.metadata?.text?.slice(0, 300) || "No text available";
        const truncationIndicator = match.metadata?.text?.length > 300 ? "..." : "";
        console.log(`🔹 Rank ${rankIndex + 1} | Similarity: ${similarityScore}%`);
        console.log(`📄 Chunk ID: ${match.id}`);
        console.log(`📝 Preview: ${previewText}${truncationIndicator}`);
        console.log("─".repeat(80) + "\n");
        return {
          id: match.id,
          similarity: match.score,
          preview: previewText,
          fullText: match.metadata?.text || ""
        };
      });

      // Use top chunks as context for GPT-4o
      const topChunks = searchResults.matches.map(match => ({
        text: match.metadata?.text || ""
      }));
      answer = await answerWithGPT4O(searchQuery, topChunks);
      console.log("💡 AI Answer:\n", answer);
    } else {
      console.log("❌ No relevant chunks found for your search query");
      answer = "I couldn't find any relevant information to answer your question.";
    }

    return {
      success: true,
      query: searchQuery,
      answer: answer,
      sources: sources
    };

  } catch (error) {
    console.error("❌ Search failed:", error.message);
    throw new Error(`Search operation failed: ${error.message}`);
  }
}
// ===== MAIN PIPELINE =====

/**
 * Executes the full RAG pipeline including index creation, text processing, embedding, and Pinecone upload
 * @param {Object} options - Pipeline options
 * @param {string} options.textFilePath - Path to the text file to process (default: SOURCE_TEXT_FILE_PATH)
 * @param {number} options.chunkSize - Size of text chunks in characters (default: 512)
 * @param {string} options.testQuery - Optional test query to run after indexing (default: "ზღვა")
 * @returns {Promise<Object>} - Result of the pipeline execution
 */
async function runRagPipeline(options = {}) {
  const {
    textFilePath = SOURCE_TEXT_FILE_PATH,
    chunkSize = 512,
    testQuery = "ზღვა"
  } = options;
  
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
    const cleanedTextContent = loadAndCleanTextFile(textFilePath);
    const textChunks = await createTextChunks(cleanedTextContent, chunkSize);

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

    // Step 5: Test search functionality and answer with GPT-4o
    console.log("\n" + "=".repeat(50));
    console.log("STEP 5: SEARCH & AI ANSWER");
    console.log("=".repeat(50));
    const searchResult = await searchAndAnswer(testQuery, 3);

    console.log("\n🎉 Pipeline completed successfully!");
    
    return {
      success: true,
      chunkCount: textChunks.length,
      testQuery: testQuery,
      testResult: searchResult
    };
    
  } catch (error) {
    console.error("\n❌ Pipeline failed:", error.message);
    console.error("Stack trace:", error.stack);
    return {
      success: false,
      error: error.message,
      stack: error.stack
    };
  }
}

// ===== EXECUTION =====

// Only run the pipeline automatically if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runRagPipeline().catch((error) => {
    console.error("💥 Unhandled error in main pipeline:", error.message);
    console.error("Full error:", error);
    process.exit(1);
  });
}

// Export functions for use in other files
export { runRagPipeline, searchAndAnswer };