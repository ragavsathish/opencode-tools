
import { type Plugin, tool } from "@opencode-ai/plugin"
import { QdrantClient } from "@qdrant/js-client-rest"
import { createHash } from "crypto"

interface RooSearchConfig {
  qdrantClientUrl: string
  embeddingServiceUrl: string
  embeddingModel: string
  searchScoreThreshold: number
  searchLimit: number
  hnswEf: number
  hnswExact: boolean
  payloadIncludeFields: string[]
  hashLength: number
}

const ROO_SEARCH_CONFIG: RooSearchConfig = {
  qdrantClientUrl: "http://localhost:6333",
  embeddingServiceUrl: "http://localhost:11434/api/embed",
  embeddingModel: "nomic-embed-text",
  searchScoreThreshold: 0.40,
  searchLimit: 50,
  hnswEf: 128,
  hnswExact: false,
  payloadIncludeFields: ["filePath", "codeChunk", "startLine", "endLine", "pathSegments"],
  hashLength: 16,
}

interface VectorStoreSearchResult {
  id: string | number
  score: number
  payload?: Payload | null
}

interface Payload {
  filePath: string
  codeChunk: string
  startLine: number
  endLine: number
  [key: string]: any
}

interface EmbeddingResponse {
  embeddings: number[][]
}

interface SearchResult {
  filePath: string
  score: number
  startLine: number
  endLine: number
  codeChunk: string
}

interface SearchResults {
  query: string
  results: readonly SearchResult[]
}

const getWorkspaceVectorStoreName = (workspacePath: string): string => {
  const hash = createHash("sha256").update(workspacePath).digest("hex")
  return `ws-${hash.substring(0, ROO_SEARCH_CONFIG.hashLength)}`
}

const qdrantClient: QdrantClient = new QdrantClient({
  url: ROO_SEARCH_CONFIG.qdrantClientUrl,
})

const createEmbeddings = async (input: string): Promise<EmbeddingResponse> => {
  const url = ROO_SEARCH_CONFIG.embeddingServiceUrl
  const model = ROO_SEARCH_CONFIG.embeddingModel

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
    console.error("Embedding creation failed:", error)
    throw new Error(`Embedding creation failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }
}

const searchQdrant = async (embedding: EmbeddingResponse, workspacePath: string): Promise<VectorStoreSearchResult[]> => {
  const searchRequest = {
    query: embedding.embeddings[0],
    filter: undefined,
    score_threshold: ROO_SEARCH_CONFIG.searchScoreThreshold,
    limit: ROO_SEARCH_CONFIG.searchLimit,
    params: {
      hnsw_ef: ROO_SEARCH_CONFIG.hnswEf,
      exact: ROO_SEARCH_CONFIG.hnswExact,
    },
    with_payload: {
      include: ROO_SEARCH_CONFIG.payloadIncludeFields,
    },
  }

  try {
    const operationResult = await qdrantClient.query(getWorkspaceVectorStoreName(workspacePath), searchRequest)
    return operationResult.points as VectorStoreSearchResult[]
  } catch (error) {
    throw new Error(`Search operation failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }
}

const filterValidResults = (results: VectorStoreSearchResult[]): VectorStoreSearchResult[] => {
  const filtered = results.filter(result => {
    const isValid = result.payload &&
      "filePath" in result.payload &&
      result.payload.filePath &&
      result.payload.startLine &&
      result.payload.endLine &&
      result.payload.codeChunk
    return isValid
  })
  return filtered
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

const executeSearchPipeline = async (query: string, workspacePath: string): Promise<string> => {
  try {
    const embeddingResponse = await createEmbeddings(query)
    const searchResults = await searchQdrant(embeddingResponse, workspacePath)
    const transformedResults = transformResults(searchResults)
    return formatResults(transformedResults, query)
  } catch (error) {
    console.error("Error executing search pipeline:", error)
    return `Error executing search: ${error instanceof Error ? error.message : 'Unknown error'}`
  }
}

export const RooSearchPlugin: Plugin = async ({directory}) => {
  return {
    tool: {
      roosearch: tool({
        description: "Query the workspace using roosearch qdrant indexing",
        args: {
          query: tool.schema.string(),
        },
        async execute(args) {
          return await executeSearchPipeline(args.query, directory)
        },
      }),
    },
  }
}


