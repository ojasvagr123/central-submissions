import { createContext, useState, useEffect } from 'react';
import { login as apiLogin, signup as apiSignup, googleAuth as apiGoogleAuth } from '../api/api';

export const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [token, setToken] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Load token from localStorage on app mount
    useEffect(() => {
        const savedToken = localStorage.getItem('token');
        const savedUser = localStorage.getItem('currentUser');
        if (savedToken) {
            setToken(savedToken);
        }
        if (savedUser) {
            try {
                setUser(JSON.parse(savedUser));
            } catch {
                setUser(null);
            }
        }
        setLoading(false);
    }, []);

    const login = async (email, password) => {
        setError(null);
        setLoading(true);
        try {
            const data = await apiLogin(email, password);
            // Backend returns: { success: true, data: { token: "...", user: {...} } }
            const authToken = data.data.token;
            setToken(authToken);
            localStorage.setItem('token', authToken);
            const userData = data.data.user || { email };
            setUser(userData);
            localStorage.setItem('currentUser', JSON.stringify(userData));
            return data;
        } catch (err) {
            const message = err?.detail?.message || err?.detail || err?.message || 'Login failed';
            setError(message);
            throw err;
        } finally {
            setLoading(false);
        }
    };

    const signup = async (name, email, password) => {
        setError(null);
        setLoading(true);
        try {
            const data = await apiSignup(name, email, password);
            // Backend returns: { success: true, data: { token: "...", user: {...} } }
            const authToken = data.data?.token;
            if (authToken) {
                setToken(authToken);
                localStorage.setItem('token', authToken);
                const userData = data.data?.user || { email };
                setUser(userData);
                localStorage.setItem('currentUser', JSON.stringify(userData));
            }
            return data;
        } catch (err) {
            const message = err?.detail?.message || err?.detail || err?.message || 'Signup failed';
            setError(message);
            throw err;
        } finally {
            setLoading(false);
        }
    };

    const googleAuth = async (id_token) => {
        setError(null);
        setLoading(true);
        try {
            const data = await apiGoogleAuth(id_token);
            // Backend returns: { success: true, data: { token: "...", user: {...} } }
            const authToken = data.data.token;
            setToken(authToken);
            localStorage.setItem('token', authToken);
            const userData = data.data.user || {};
            setUser(userData);
            localStorage.setItem('currentUser', JSON.stringify(userData));
            return data;
        } catch (err) {
            const message = err?.detail?.message || err?.detail || err?.message || 'Google auth failed';
            setError(message);
            throw err;
        } finally {
            setLoading(false);
        }
    };

    const logout = () => {
        setUser(null);
        setToken(null);
        setError(null);
        localStorage.removeItem('token');
        localStorage.removeItem('currentUser');
    };

    const value = {
        user,
        token,
        loading,
        error,
        login,
        signup,
        googleAuth,
        logout,
    };

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
};
