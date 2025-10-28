import { tool } from "@opencode-ai/plugin"
import { QdrantClient } from "@qdrant/js-client-rest"

export interface VectorStoreSearchResult {
  id: string | number
  score: number
  payload?: Payload | null
}

export interface Payload {
  filePath: string
  codeChunk: string
  startLine: number
  endLine: number
  [key: string]: any
}

export interface EmbeddingResponse {
  embeddings: number[][]
}

export interface SearchResult {
  filePath: string
  score: number
  startLine: number
  endLine: number
  codeChunk: string
}

export interface SearchResults {
  query: string
  results: readonly SearchResult[]
}

export const qdrantClient: QdrantClient = new QdrantClient({
  url: "http://localhost:6333",
})


const createEmbeddings = async (input: string): Promise<EmbeddingResponse> => {
  const url = "http://localhost:11434/api/embed"
  const model = "nomic-embed-text"
  
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        input
      }),
    })

    if (!response.ok) {
      throw new Error(`Failed to create embeddings: ${response.statusText}`)
    }

    const data = await response.json() as EmbeddingResponse
    return data
  } catch (error) {
    throw new Error(`Embedding creation failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }
}


export const searchQdrant = async (embedding: EmbeddingResponse): Promise<VectorStoreSearchResult[]> => {
  const searchRequest = {
    query: embedding.embeddings[0],
    filter: undefined,
    score_threshold: .40,
    limit: 50,
    params: {
      hnsw_ef: 128,
      exact: false,
    },
    with_payload: {
      include: ["filePath", "codeChunk", "startLine", "endLine", "pathSegments"],
    },
  }
  
  try {
    const operationResult = await qdrantClient.query("ws-e9be3207f7247f18", searchRequest)
    return operationResult.points as VectorStoreSearchResult[]
  } catch (error) {
    throw new Error(`Search operation failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }
}


const filterValidResults = (results: VectorStoreSearchResult[]): VectorStoreSearchResult[] => {
  return results.filter(result => 
    result.payload && 
    "filePath" in result.payload && 
    result.payload.filePath &&
    result.payload.startLine &&
    result.payload.endLine &&
    result.payload.codeChunk
  )
}

const transformSearchResult = (result: VectorStoreSearchResult): SearchResult | null => {
  if (!result.payload || !("filePath" in result.payload)) {
    return null
  }
  
  return {
    filePath: result.payload.filePath,
    score: result.score,
    startLine: result.payload.startLine,
    endLine: result.payload.endLine,
    codeChunk: result.payload.codeChunk.trim(),
  }
}

const transformResults = (results: VectorStoreSearchResult[]): readonly SearchResult[] => {
  const filteredResults = filterValidResults(results)
  const transformed = filteredResults
    .map(transformSearchResult)
    .filter((result): result is SearchResult => result !== null)
  
  return transformed as readonly SearchResult[]
}

const formatSearchResult = (result: SearchResult): string => {
  return `File: ${result.filePath}
Score: ${result.score.toFixed(3)}
Lines: ${result.startLine}-${result.endLine}
Code:
\`\`\`
${result.codeChunk}
\`\`\`
`
}

const formatResults = (results: readonly SearchResult[], query: string): string => {
  if (results.length === 0) {
    return `Query: "${query}"

No results found.`
  }

  const resultsArray = Array.from(results)
  const sortedResults = [...resultsArray].sort((a: SearchResult, b: SearchResult) => b.score - a.score)
  
  const formattedResults = sortedResults
    .map(formatSearchResult)
    .join('\n\n')

  return `Query: "${query}"

Found ${results.length} result${results.length !== 1 ? 's' : ''}:

${formattedResults}

---
Search completed with ${results.length} result${results.length !== 1 ? 's' : ''}.`
}

const executeSearchPipeline = async (query: string): Promise<string> => {
  try {
    const embeddingResponse = await createEmbeddings(query)
    const searchResults = await searchQdrant(embeddingResponse)
    const transformedResults = transformResults(searchResults)
    return formatResults(transformedResults, query)
  } catch (error) {
    return `Error executing search: ${error instanceof Error ? error.message : 'Unknown error'}`
  }
}

export default tool({
  description: "Query the cuent workspace using roo code qdrant indexing",
  args: {
    query: tool.schema.string().describe("Search query string"),
  },
  async execute(args) {
    return await executeSearchPipeline(args.query)
  },
})