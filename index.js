import fs from "fs";
import { pipeline } from "@xenova/transformers";
import { Pinecone } from "@pinecone-database/pinecone";

// --- Configuration Constants ---
const TEXT_FILE = "./Dummy Text Files/corpora.txt"; // Path to your source text file
const CHUNK_SIZE = 500; // Size of text chunks (in words)
const OVERLAP = 50; // Overlap between chunks (in words)

// Pinecone Configuration
// IMPORTANT: Replace these with your actual Pinecone API Key and Controller Host URL.
// For production, ALWAYS use environment variables (e.g., process.env.PINECONE_API_KEY).
const PINECONE_API_KEY =
  "pcsk_TqZYd_2JKFQdA9hNpduqVHPx2E6Xo5LfQsLNZFRjXMDu4jnWWrtdpNkitXNs96cTHWUec"; // Your actual API Key

// YOU MUST FIND THIS URL IN YOUR PINECONE DASHBOARD under 'API Keys' -> 'Controller Host'
// It looks like: "https://controller.YOUR_REGION.pinecone.io" (e.g., "https://controller.us-east-1.pinecone.io")
const PINECONE_CONTROLLER_HOST_URL = "https://controller.us-east-1.pinecone.io"; // <--- REPLACE WITH YOUR ACTUAL CONTROLLER HOST URL

const PINECONE_INDEX_NAME = "my-bge-m3-index"; // The name of your Pinecone index
const BGE_M3_DIMENSION = 1024; // The output dimension of 'Xenova/bge-m3' model

// The region of your Pinecone index. This is needed for index creation.
// Based on your previous screenshot, this is "us-east-1".
const PINECONE_INDEX_REGION = "us-east-1";

// Initialize Pinecone client globally
const pc = new Pinecone({
  apiKey: PINECONE_API_KEY,
  controllerHostUrl: PINECONE_CONTROLLER_HOST_URL, // This is the ONLY correct host parameter
});

// ---- Step 1: Load and clean raw text ----
/**
 * Loads text from a file and performs basic cleaning.
 * @param {string} filePath - The path to the text file.
 * @returns {string} The cleaned text.
 */
function loadAndCleanText(filePath) {
  try {
    const rawText = fs.readFileSync(filePath, "utf-8");
    return rawText.replace(/\s+/g, " ").trim(); // Replace multiple whitespaces with single space and trim
  } catch (error) {
    console.error(`❌ Error loading text file ${filePath}:`, error);
    process.exit(1); // Exit if file cannot be read
  }
}

// ---- Step 2: Chunk text with overlap ----
/**
 * Chunks the text into smaller pieces with specified overlap.
 * @param {string} text - The input text.
 * @param {number} chunkSize - The maximum number of words per chunk.
 * @param {number} overlap - The number of words to overlap between chunks.
 * @returns {Array<Object>} An array of chunk objects with id and text.
 */
function chunkText(text, chunkSize = 500, overlap = 50) {
  const words = text.split(" ");
  const chunks = [];
  for (let i = 0; i < words.length; i += chunkSize - overlap) {
    const chunkWords = words.slice(i, i + chunkSize);
    chunks.push({
      id: `chunk_${chunks.length}`, // Pinecone requires string IDs
      text: chunkWords.join(" "),
    });
  }
  return chunks;
}

// ---- Step 3: Embed text using Xenova/bge-m3 ----
/**
 * Embeds an array of text chunks using the Xenova/bge-m3 model.
 * @param {Array<Object>} chunks - An array of chunk objects.
 * @returns {Promise<Array<Float32Array>>} A promise that resolves to an array of embeddings.
 */
async function embedChunks(chunks) {
  console.log("🧠 Initializing embedding model 'Xenova/bge-m3'...");
  const embedder = await pipeline("feature-extraction", "Xenova/bge-m3");
  console.log("✅ Embedding model loaded.");

  const embeddings = [];
  for (let i = 0; i < chunks.length; i++) {
    if (i % 10 === 0 || i === chunks.length - 1) {
      console.log(`Embedding chunk ${i + 1} of ${chunks.length}...`);
    }
    // Perform embedding and normalize the output
    const output = await embedder(chunks[i].text, {
      pooling: "cls",
      normalize: true,
    });
    embeddings.push(output.data); // output.data is a Float32Array
  }
  return embeddings;
}

// ---- Step 4: Save chunks and embeddings to Pinecone ----
/**
 * Upserts text chunks and their embeddings into a Pinecone index.
 * The text content is stored in the metadata.
 * @param {Array<Object>} chunks - The array of chunk objects.
 * @param {Array<Float32Array>} embeddings - The array of corresponding embeddings.
 */
async function saveToPinecone(chunks, embeddings) {
  const index = pc.Index(PINECONE_INDEX_NAME);
  const vectorsToUpsert = [];

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const embedding = embeddings[i];

    // Prepare vector object for Pinecone upsert
    vectorsToUpsert.push({
      id: chunk.id, // Unique string ID for the vector
      values: Array.from(embedding), // Convert Float32Array to regular Array for serialization
      metadata: {
        text: chunk.text, // Store the original text chunk in metadata
        original_chunk_id: chunk.id, // Optional: for additional reference
      },
    });
  }

  // Upsert vectors to Pinecone in batches for efficiency
  const batchSize = 100; // Pinecone recommends batches for upserts
  console.log(`💾 Starting upsert to Pinecone in batches of ${batchSize}...`);
  for (let i = 0; i < vectorsToUpsert.length; i += batchSize) {
    const batch = vectorsToUpsert.slice(i, i + batchSize);
    try {
      await index.upsert(batch);
      console.log(
        `✅ Upserted batch ${Math.ceil(
          (i + batchSize) / batchSize
        )} of ${Math.ceil(vectorsToUpsert.length / batchSize)}`
      );
    } catch (error) {
      console.error(`❌ Error upserting batch starting at index ${i}:`, error);
      // In a real application, you might implement retry logic or more robust error handling
      throw error; // Re-throw to halt if a batch fails completely
    }
  }

  console.log(
    `🎉 Successfully uploaded ${vectorsToUpsert.length} vectors and chunks to Pinecone index: ${PINECONE_INDEX_NAME}`
  );
}

// ---- Step 5: Query Embedding & Search from Pinecone ----
/**
 * Embeds a query and searches the Pinecone index for similar chunks.
 * @param {string} query - The query string.
 * @param {number} topK - The number of top similar results to retrieve.
 */
async function searchQuery(query, topK = 5) {
  const index = pc.Index(PINECONE_INDEX_NAME);
  console.log(`🧠 Embedding query: "${query}"...`);
  const embedder = await pipeline("feature-extraction", "Xenova/bge-m3");
  const queryEmbedding = (
    await embedder(query, { pooling: "cls", normalize: true })
  ).data;
  console.log("✅ Query embedded.");

  console.log(
    `🔎 Searching Pinecone index '${PINECONE_INDEX_NAME}' for similar vectors...`
  );
  try {
    const queryResult = await index.query({
      vector: Array.from(queryEmbedding), // Convert to array for Pinecone query
      topK: topK,
      includeMetadata: true, // Crucial: tells Pinecone to return the stored metadata (your text chunks)
    });

    if (queryResult.matches && queryResult.matches.length > 0) {
      console.log(`✨ Found ${queryResult.matches.length} relevant results:`);
      for (let rank = 0; rank < queryResult.matches.length; rank++) {
        const match = queryResult.matches[rank];
        console.log(
          `\n🔹 Rank ${rank + 1} | Chunk ID: ${
            match.id
          } | Score: ${match.score.toFixed(4)}`
        );
        // Check if metadata.text exists before accessing
        if (match.metadata && match.metadata.text) {
          console.log(
            match.metadata.text.slice(0, 300) +
              (match.metadata.text.length > 300 ? "..." : ""),
            "\n---"
          );
        } else {
          console.log("No text found in metadata for this chunk.", "\n---");
        }
      }
    } else {
      console.log("No relevant chunks found in Pinecone for your query.");
    }
  } catch (error) {
    console.error("❌ Error during Pinecone search:", error);
  }
}

// ---- Main Pipeline ----
async function main() {
  console.log("🚀 Initializing Pinecone client and ensuring index exists...");
  try {
    // Check if the index already exists in your Pinecone account
    const indexList = await pc.listIndexes();
    if (!indexList.includes(PINECONE_INDEX_NAME)) {
      console.log(`Creating Pinecone index '${PINECONE_INDEX_NAME}'...`);
      // Create the index with bge-m3 compatible settings
      await pc.createIndex({
        name: PINECONE_INDEX_NAME,
        dimension: BGE_M3_DIMENSION, // Must match your embedding model's output dimension
        metric: "cosine", // bge-m3 uses cosine similarity
        spec: {
          // Define the serverless specific details
          serverless: {
            cloud: "aws", // Specify your cloud provider (e.g., 'aws', 'gcp', 'azure')
            region: PINECONE_INDEX_REGION, // Use the region where you want the index to be
          },
        },
      });
      console.log(`✅ Pinecone index '${PINECONE_INDEX_NAME}' created.`);
    } else {
      console.log(`✅ Pinecone index '${PINECONE_INDEX_NAME}' already exists.`);
    }
  } catch (error) {
    console.error("❌ Failed to initialize Pinecone or ensure index:", error);
    process.exit(1); // Exit if critical Pinecone setup fails
  }

  console.log("\n📄 Loading and preparing text from local file...");
  const text = loadAndCleanText(TEXT_FILE);
  const chunks = chunkText(text, CHUNK_SIZE, OVERLAP);
  console.log(`✅ Text split into ${chunks.length} chunks.`);

  console.log("\n🧠 Starting embedding process for all chunks...");
  const embeddings = await embedChunks(chunks);

  console.log("\n💾 Saving chunks and embeddings to Pinecone...");
  await saveToPinecone(chunks, embeddings);

  console.log("\n🎉 Data ingestion complete!");

  console.log("\n--- Testing Search Queries ---");
  // Test search queries
  await searchQuery("რას ამბობს პლატონი სამართლიანობაზე"); // Example in Georgian
  await searchQuery("What are the core ideas of Plato?"); // Example in English
  await searchQuery("What is justice according to philosophers?"); // Another example

  console.log("\n--- End of Program ---");
}

// Run the main pipeline and catch any top-level errors
main().catch((error) => {
  console.error("An unhandled error occurred in the main pipeline:", error);
  process.exit(1); // Exit with an error code
});
