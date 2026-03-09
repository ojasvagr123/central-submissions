# Prashikshan AI — Frontend

React frontend for the **Prashikshan AI Mock Interview Platform**. Provides an interactive interface for candidates to practice technical interviews powered by a FastAPI + AI backend.

## 🚀 Tech Stack

| Layer | Technology |
|---|---|
| UI Library | React 19 |
| Build Tool | Vite (Rolldown) |
| HTTP Client | Axios |
| Styling | Tailwind CSS v4 |
| Routing | React Router v7 |
| State | React Context API + Custom Hooks |

## 📁 Project Structure

```
src/
├── api/
│   └── api.js               # Axios instance, interceptors, all API functions
├── Components/
│   ├── ProtectedRoute.jsx    # Auth guard for private routes
│   └── QuestionCard.jsx      # Question display + answer submission + feedback
├── context/
│   └── AuthContext.jsx       # Global auth state (user, token, login/signup/logout)
├── hooks/
│   ├── useAuth.js            # Shortcut hook for AuthContext
│   └── useInterview.js       # Interview flow state machine (start → answer → summary)
├── pages/
│   ├── LoginPage.jsx         # Login with email/password or Google OAuth
│   ├── SignupPage.jsx        # Registration with validation
│   ├── InterviewConfig.jsx   # Subject, bloom level, difficulty, question count
│   └── InterviewScreen.jsx   # Live interview: timer, progress, Q&A flow
├── App.jsx                   # Routes and AuthProvider
└── main.jsx                  # Entry point
```

## ✨ Features

- **JWT Authentication** — Login, signup, and Google OAuth. Token stored in `localStorage` and auto-attached via axios interceptor. Auto-redirect on 401.
- **Protected Routes** — `/config` and `/interview` require authentication.
- **Interview Configuration** — Choose subject, Bloom's taxonomy level (single or mixed), difficulty, and number of questions (1–20).
- **Live Interview Flow** — AI-generated questions fetched one at a time. Countdown timer, progress bar, Bloom level and difficulty badges.
- **Answer Submission & Feedback** — Submit answers inline, receive AI-evaluated score and feedback before moving to the next question.
- **Session Summary** — Auto-fetched on completion with average score, performance level, and per-question breakdown.

## 🛠️ Setup

### Prerequisites
- Node.js ≥ 16.14
- npm

### Install & Run

```bash
# Install dependencies
npm install

# Start dev server (default: http://localhost:5173)
npm run dev
```

### Backend Connection

The API base URL is set in `src/api/api.js`:
```js
const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';
```

The FastAPI backend must be running for full functionality. CORS is pre-configured for `localhost:5173`.

## 🔗 API Endpoints

All routes are prefixed with `/api/v1`.

### Authentication (no token required)

| Method | Route | Request Body | Response |
|---|---|---|---|
| `POST` | `/auth/signup` | `{ name, email, password }` | `{ success, data: { token, user } }` |
| `POST` | `/auth/login` | `{ email, password }` | `{ success, data: { token, user } }` |
| `POST` | `/auth/google` | `{ id_token }` | `{ success, data: { token, user } }` |

### Interview (Bearer token required)

| Method | Route | Request Body | Response |
|---|---|---|---|
| `POST` | `/interview/start` | `{ subject, mode, bloom_level, difficulty, num_questions, language, bloom_strategy }` | `{ session_id, questions[] }` |
| `GET` | `/interview/{id}/next` | — | `{ status, interview_question_id, question_text, bloom_level, sequence, time_limit }` |
| `POST` | `/interview/{id}/answer` | `{ interview_question_id, user_answer }` | `{ score, feedback, insights }` |
| `GET` | `/interview/{id}/summary` | — | `{ average_score, performance_level, total_answered, breakdown[] }` |

## 📜 Scripts

| Command | Description |
|---|---|
| `npm run dev` | Start Vite dev server |
| `npm run build` | Production build |
| `npm run lint` | Run ESLint |
| `npm run preview` | Preview production build |
