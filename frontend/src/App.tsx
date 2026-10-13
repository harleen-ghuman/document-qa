import { useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type Evidence = {
  source: string
  page?: number
  text: string
}

type Citation = {
  source: string
  page?: number
}

type Answer = {
  question: string
  answer: string
  citations?: Citation[]
  retrieved_evidence?: Evidence[]
}

type AnswerResponse = {
  results: Answer[]
}

export default function App() {
  const [documentFile, setDocumentFile] = useState<File | null>(null)
  const [questionsFile, setQuestionsFile] = useState<File | null>(null)
  const [mode, setMode] = useState<'typed' | 'file'>('typed')
  const [question, setQuestion] = useState('')
  const [results, setResults] = useState<Answer[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (loading) return

    setError('')
    setResults([])

    if (!documentFile) {
      setError('Please upload a source document.')
      return
    }

    if (mode === 'typed' && !question.trim()) {
      setError('Please enter a question.')
      return
    }

    if (mode === 'file' && !questionsFile) {
      setError('Please upload a questions JSON file.')
      return
    }

    const data = new FormData()
    data.append('document_file', documentFile)

    if (mode === 'typed') {
      const questions = new Blob(
          [JSON.stringify({ questions: [question.trim()] })],
          { type: 'application/json' },
      )
      data.append('questions_file', questions, 'questions.json')
    } else {
      data.append('questions_file', questionsFile!)
    }

    setLoading(true)

    try {
      const response = await fetch('/api/answer?include_evidence=true', {
        method: 'POST',
        body: data,
      })

      if (!response.ok) {
        const body = await response.json().catch(() => null)
        throw new Error(
            typeof body?.detail === 'string'
                ? body.detail
                : `Request failed (${response.status}). Please try again.`,
        )
      }

      const body: AnswerResponse = await response.json()
      setResults(body.results)
    } catch (err) {
      setError(
          err instanceof Error ? err.message : 'Unable to complete the request.',
      )
    } finally {
      setLoading(false)
    }
  }

  return (
      <main className="container">
        <header>
          <span className="badge">Zania · Document Q&A</span>
          <h1>Ask your document.</h1>
          <p>Upload a PDF or JSON document and get answers with evidence.</p>
        </header>

        <form className="panel" onSubmit={handleSubmit}>
          <fieldset disabled={loading}>
            <label htmlFor="document">Source document</label>
            <input
                id="document"
                type="file"
                accept=".pdf,.json"
                required
                onChange={(event) =>
                    setDocumentFile(event.target.files?.[0] ?? null)
                }
            />

            <label htmlFor="mode">How would you like to ask?</label>
            <select
                id="mode"
                value={mode}
                onChange={(event) =>
                    setMode(event.target.value as 'typed' | 'file')
                }
            >
              <option value="typed">Type one question</option>
              <option value="file">Upload a questions file</option>
            </select>

            {mode === 'typed' ? (
                <>
                  <label htmlFor="question">Your question</label>
                  <textarea
                      id="question"
                      value={question}
                      onChange={(event) => setQuestion(event.target.value)}
                      placeholder="Which cloud providers does the service use?"
                      maxLength={1000}
                      required
                  />
                </>
            ) : (
                <>
                  <label htmlFor="questions">Questions JSON file</label>
                  <input
                      id="questions"
                      type="file"
                      accept=".json"
                      required
                      onChange={(event) =>
                          setQuestionsFile(event.target.files?.[0] ?? null)
                      }
                  />
                  <p className="hint">
                    Use a JSON object with a “questions” list or a bare list of strings.
                    Maximum 20 questions.
                  </p>
                </>
            )}

            <button type="submit">
              {loading ? 'Searching…' : 'Get answers'}
            </button>
          </fieldset>
        </form>

        <div role="status" aria-live="polite">
          {loading && (
              <p>Searching the document…</p>
          )}
          {!loading && results.length > 0 && (
              <p>{results.length} answer(s) returned. Review sources to verify them.</p>
          )}
        </div>

        {error && <p className="error" role="alert">{error}</p>}

        <section aria-label="Answers" aria-busy={loading}>
          {results.map((result, index) => (
              <article className="panel" key={index}>
                <h2>{result.question}</h2>
                <p className="answer">{result.answer}</p>
                {!!result.citations?.length && (
                    <div className="citations">
                      <strong>Cited sources:</strong>
                      <ul>
                        {result.citations.map((citation, citationIndex) => (
                            <li key={citationIndex}>
                              {citation.source}
                              {citation.page != null && ` · Page ${citation.page}`}
                            </li>
                        ))}
                      </ul>
                    </div>
                )}

                {!!result.retrieved_evidence?.length && (
                    <details>
                      <summary>View retrieved sources</summary>
                      {result.retrieved_evidence.map((evidence, evidenceIndex) => (
                          <div className="evidence" key={evidenceIndex}>
                            <strong>
                              {evidence.source}
                              {evidence.page != null && ` · Page ${evidence.page}`}
                            </strong>
                            <pre>{evidence.text}</pre>
                          </div>
                      ))}
                    </details>
                )}
              </article>
          ))}
        </section>
      </main>
  )
}