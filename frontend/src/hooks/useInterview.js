import { useState } from 'react';
import {
    startInterview as apiStartInterview,
    getNextQuestion as apiGetNextQuestion,
    submitAnswer as apiSubmitAnswer,
    getSummary as apiGetSummary,
} from '../api/api';

const useInterview = () => {
    const [sessionId, setSessionId] = useState(null);
    const [currentQuestion, setCurrentQuestion] = useState(null);
    const [loading, setLoading] = useState(false);
    const [answers, setAnswers] = useState([]);
    const [summary, setSummary] = useState(null);
    const [error, setError] = useState(null);
    const [isComplete, setIsComplete] = useState(false);
    const [questionNumber, setQuestionNumber] = useState(0);

    const start = async (payload) => {
        setError(null);
        setLoading(true);
        setSummary(null);
        setAnswers([]);
        setIsComplete(false);
        setQuestionNumber(0);
        try {
            // Backend returns raw: { session_id, questions: [...] }
            const res = await apiStartInterview(payload);
            const sid = res.session_id;
            setSessionId(sid);
            return sid;
        } catch (err) {
            const message = err?.detail || err?.message || 'Failed to start interview';
            setError(message);
            throw err;
        } finally {
            setLoading(false);
        }
    };

    const fetchNextQuestion = async (sid) => {
        const id = sid || sessionId;
        if (!id) return null;

        setError(null);
        setLoading(true);
        try {
            // Backend returns raw: { status, interview_question_id, question_text, bloom_level, sequence, time_limit }
            // or { status: "completed", message: "..." }
            const data = await apiGetNextQuestion(id);

            // If backend signals no more questions
            if (!data || data.status === 'completed') {
                setIsComplete(true);
                setCurrentQuestion(null);
                // Auto-fetch summary
                await fetchSummary(id);
                return null;
            }

            setCurrentQuestion(data);
            setQuestionNumber((prev) => prev + 1);
            return data;
        } catch (err) {
            // Backend returns 404 when session not found
            if (err?.status === 404 || err?.detail?.includes?.('no more')) {
                setIsComplete(true);
                setCurrentQuestion(null);
                await fetchSummary(id);
                return null;
            }
            const message = err?.detail || err?.message || 'Failed to get next question';
            setError(message);
            throw err;
        } finally {
            setLoading(false);
        }
    };

    const answer = async (interview_question_id, user_answer) => {
        if (!sessionId) return;

        setError(null);
        setLoading(true);
        try {
            // Backend returns raw: { score, feedback, insights }
            const data = await apiSubmitAnswer(sessionId, interview_question_id, user_answer);
            setAnswers((prev) => [...prev, { interview_question_id, user_answer, response: data }]);
            return data;
        } catch (err) {
            const message = err?.detail || err?.message || 'Failed to submit answer';
            setError(message);
            throw err;
        } finally {
            setLoading(false);
        }
    };

    const fetchSummary = async (sid) => {
        const id = sid || sessionId;
        if (!id) return;

        setError(null);
        setLoading(true);
        try {
            // Backend returns raw: { average_score, performance_level, total_answered, breakdown }
            const data = await apiGetSummary(id);
            setSummary(data);
            setIsComplete(true);
            return data;
        } catch (err) {
            const message = err?.detail || err?.message || 'Failed to get summary';
            setError(message);
            throw err;
        } finally {
            setLoading(false);
        }
    };

    const reset = () => {
        setSessionId(null);
        setCurrentQuestion(null);
        setLoading(false);
        setAnswers([]);
        setSummary(null);
        setError(null);
        setIsComplete(false);
        setQuestionNumber(0);
    };

    return {
        sessionId,
        currentQuestion,
        loading,
        answers,
        summary,
        error,
        isComplete,
        questionNumber,
        start,
        fetchNextQuestion,
        answer,
        fetchSummary,
        reset,
    };
};

export default useInterview;
