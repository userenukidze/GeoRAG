import express from 'express';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import { searchAndAnswer } from './index.js';

const app = express();
const port = process.env.PORT || 5000;

// Setup file storage
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/');
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

// Create uploads directory if it doesn't exist
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

const upload = multer({ storage: storage });

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// API endpoint to handle form submissions
app.post('/api/ask', upload.array('files'), async (req, res) => {
  console.log('\n========== FORM SUBMISSION RECEIVED ==========');
  console.log('📝 PROMPT:', req.body.prompt);

  if (req.files && req.files.length > 0) {
    console.log(`📁 FILES RECEIVED: ${req.files.length}`);
    req.files.forEach((file, index) => {
      console.log(`  📄 File ${index + 1}: ${file.originalname} (${file.size} bytes)`);
      console.log(`     Saved as: ${file.filename}`);
    });
  } else {
    console.log('📁 NO FILES RECEIVED');
  }
  console.log('============================================\n');

  try {
    // Pass request info to searchAndAnswer
    const requestInfo = {
      source: 'getFile.js',
      files: req.files || []
    };
    const result = await searchAndAnswer(req.body.prompt, 3, { requestInfo });

    res.status(200).json({
      success: true,
      answer: result.answer,
      sources: result.sources,
      message: "Query processed successfully"
    });
  } catch (error) {
    console.error("❌ Error processing query:", error.message);
    res.status(500).json({
      success: false,
      message: "Failed to process your query",
      error: error.message
    });
  }
});

// Start server
app.listen(port, () => {
  console.log(`🚀 Server running on port ${port}`);
  console.log(`🌐 API endpoint available at http://localhost:${port}/api/ask`);
});