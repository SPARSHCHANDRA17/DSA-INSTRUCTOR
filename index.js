import * as dotenv from "dotenv";

dotenv.config();

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";

async function indexDocument() {
  try {
    // ==========================================
    // 1. LOAD PDF
    // ==========================================

    const PDF_PATH = "./dsa.pdf";

    const pdfLoader = new PDFLoader(PDF_PATH);
    const rawDocs = await pdfLoader.load();

    console.log("PDF loaded");


    // ==========================================
    // 2. CHUNK PDF
    // ==========================================

    const textSplitter =
      new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
      });

    const chunkedDocs =
      await textSplitter.splitDocuments(rawDocs);

    console.log("Chunking Completed");


    // ==========================================
    // 3. REMOVE EMPTY CHUNKS
    // ==========================================

    const validDocs = chunkedDocs.filter(
      (doc) =>
        doc.pageContent &&
        doc.pageContent.trim().length > 0
    );

    console.log(
      `Cleaned ${validDocs.length} valid chunks ready for embedding.`
    );

    if (validDocs.length === 0) {
      throw new Error(
        "No valid text content found in PDF."
      );
    }


    // ==========================================
    // 4. GEMINI EMBEDDING MODEL
    // ==========================================

    const embeddings =
      new GoogleGenerativeAIEmbeddings({});

    console.log(
      "Embedding model configured"
    );


    // ==========================================
    // 5. GENERATE EMBEDDINGS ONE BY ONE
    // ==========================================

    console.log(
      "Generating embeddings..."
    );

    const vectors = [];

    for (
      let i = 0;
      i < validDocs.length;
      i++
    ) {

      const text =
        validDocs[i].pageContent;

      const vector =
        await embeddings.embedQuery(text);

      console.log(
        `Embedding ${i + 1}/${validDocs.length} - dimension: ${vector.length}`
      );

      if (
        !Array.isArray(vector) ||
        vector.length !== 3072
      ) {
        throw new Error(
          `Invalid embedding at index ${i}. Dimension: ${vector.length}`
        );
      }

      vectors.push(vector);
    }

    console.log(
      `Generated ${vectors.length} embeddings`
    );

    console.log(
      `Embedding dimension: ${vectors[0].length}`
    );


    // ==========================================
    // 6. CONFIGURE PINECONE
    // ==========================================

    const pinecone = new Pinecone({
      apiKey:
        process.env.PINECONE_API_KEY,
    });

    const pineconeIndex =
      pinecone.Index(
        process.env.PINECONE_INDEX_NAME
      );

    console.log(
      "Pinecone configured"
    );


    // ==========================================
    // 7. CREATE PINECONE RECORDS
    // ==========================================

    const records = vectors.map(
      (vector, i) => ({
        id: `chunk-${i}`,

        values: vector,

        metadata: {
          text:
            validDocs[i].pageContent,

          source:
            validDocs[i].metadata
              ?.source || PDF_PATH,

          page:
            validDocs[i].metadata
              ?.loc?.pageNumber || 1,
        },
      })
    );

    console.log(
      `Prepared ${records.length} records for Pinecone`
    );


    // ==========================================
    // 8. UPLOAD IN BATCHES
    // ==========================================

    const batchSize = 50;

    const totalBatches =
      Math.ceil(
        records.length /
          batchSize
      );

    for (
      let i = 0;
      i < records.length;
      i += batchSize
    ) {

      const batch =
        records.slice(
          i,
          i + batchSize
        );

      const batchNumber =
        Math.floor(
          i / batchSize
        ) + 1;

      console.log(
        `Uploading batch ${batchNumber}/${totalBatches} (${batch.length} vectors)...`
      );

      await pineconeIndex.upsert(
        batch
      );

      console.log(
        `Batch ${batchNumber} uploaded successfully`
      );
    }


    // ==========================================
    // 9. SUCCESS
    // ==========================================

    console.log(
      "================================="
    );

    console.log(
      "Data stored successfully in Pinecone!"
    );

    console.log(
      `Total vectors stored: ${records.length}`
    );

    console.log(
      "================================="
    );

  } catch (error) {

    console.error(
      "================================="
    );

    console.error(
      "Error while indexing document:"
    );

    console.error(error);

    console.error(
      "================================="
    );
  }
}

indexDocument();