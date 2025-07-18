const fs = require("fs");
const { pipeline } = require("@xenova/transformers"); // npm install @xenova/transformers
const { cosineSimilarity } = require("ml-distance"); // npm install ml-distance

const TEXT_FILE = "./Dummy Text Files/corpora.txt";
const CHUNK_JSON_FILE = "chunks.json";
const EMBEDDING_FILE = "embeddings.json";
const CHUNK_SIZE = 500;
const OVERLAP = 50;

// ---- Step 1: Load and clean raw text ----
function loadAndCleanText(filePath) {
  const rawText = fs.readFileSync(filePath, "utf-8");
  return rawText.replace(/\s+/g, " ").trim();
}

// ---- Step 2: Chunk text with overlap ----
function chunkText(text, chunkSize = 500, overlap = 50) {
  const words = text.split(" ");
  const chunks = [];
  for (let i = 0; i < words.length; i += chunkSize - overlap) {
    const chunkWords = words.slice(i, i + chunkSize);
    chunks.push({
      id: chunks.length,
      text: chunkWords.join(" "),
    });
  }
  return chunks;
}

// ---- Step 3: Embed text using Xenova/bge-m3 ----
async function embedChunks(chunks) {
  const embedder = await pipeline("feature-extraction", "Xenova/bge-m3");
  const embeddings = [];
  for (let i = 0; i < chunks.length; i++) {
    if (i % 10 === 0 || i === chunks.length - 1) {
      console.log(`Embedding chunk ${i + 1} of ${chunks.length}...`);
    }
    const output = await embedder(chunks[i].text, {
      pooling: "cls",
      normalize: true,
    });
    embeddings.push(output.data);
  }
  return embeddings;
}

// ---- Step 4: Save chunks and embeddings ----
function saveChunks(chunks, filename = CHUNK_JSON_FILE) {
  fs.writeFileSync(filename, JSON.stringify(chunks, null, 2), "utf-8");
  console.log(`✅ Chunks saved to ${filename}`);
}
function saveEmbeddings(embeddings, filename = EMBEDDING_FILE) {
  fs.writeFileSync(filename, JSON.stringify(embeddings), "utf-8");
  console.log(`✅ Embeddings saved to ${filename}`);
}

// ---- Step 5: Query Embedding & Search ----
async function searchQuery(query, topK = 5) {
  const chunks = JSON.parse(fs.readFileSync(CHUNK_JSON_FILE, "utf-8"));
  const embeddings = JSON.parse(fs.readFileSync(EMBEDDING_FILE, "utf-8"));
  const embedder = await pipeline("feature-extraction", "Xenova/bge-m3");
  const queryEmbedding = (
    await embedder(query, { pooling: "cls", normalize: true })
  ).data;

  // Compute cosine similarity
  const similarities = embeddings.map((e) =>
    cosineSimilarity(e, queryEmbedding)
  );
  const topIndices = similarities
    .map((sim, idx) => ({ sim, idx }))
    .sort((a, b) => b.sim - a.sim)
    .slice(0, topK)
    .map((obj) => obj.idx);

  for (let rank = 0; rank < topIndices.length; rank++) {
    const idx = topIndices[rank];
    console.log(`🔹 Rank ${rank + 1} | Chunk ID: ${chunks[idx].id}`);
    console.log(chunks[idx].text.slice(0, 300), "\n---\n");
  }
}

// ---- Main Pipeline ----
async function main() {
  console.log("📄 Loading and preparing text...");
  const text = loadAndCleanText(TEXT_FILE);
  const chunks = chunkText(text, CHUNK_SIZE, OVERLAP);
  console.log(`✅ Text split into ${chunks.length} chunks.`);

  console.log("🧠 Starting embedding process...");
  const embeddings = await embedChunks(chunks);

  console.log("💾 Saving chunks...");
  saveChunks(chunks);
  console.log("💾 Saving embeddings...");
  saveEmbeddings(embeddings);

  console.log("🎉 Done! Chunks and embeddings saved.");
  // Uncomment to test search:
  // await searchQuery("რას ამბობს პლატონი სამართლიანობაზე");
}

main();
