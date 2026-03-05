import { useState } from 'react';
import logoImg from '../assets/logo.png';
import bgImg from '../assets/bg.png';
import { Link, useNavigate } from 'react-router-dom';
import useAuth from '../hooks/useAuth';

export default function SignupPage() {
    const [focusedField, setFocusedField] = useState(null);
    const [isHovering, setIsHovering] = useState(false);
    const [formData, setFormData] = useState({
        fullName: '',
        email: '',
        password: ''
    });
    const [signupError, setSignupError] = useState(null);
    const [signupLoading, setSignupLoading] = useState(false);
    const navigate = useNavigate();
    const { signup, googleAuth } = useAuth();

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.id]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setSignupError(null);
        setSignupLoading(true);
        try {
            await signup(formData.fullName, formData.email, formData.password);
            alert('Account created successfully! Please login.');
            navigate('/');
        } catch (err) {
            const message = err?.detail || err?.message || 'Signup failed. Please try again.';
            setSignupError(typeof message === 'string' ? message : 'Signup failed. Please try again.');
        } finally {
            setSignupLoading(false);
        }
    };

    const handleGoogleAuth = async () => {
        // Google Sign-In would provide the id_token via Google's SDK
        // For now, this is a placeholder that you can integrate with Google Identity Services
        setSignupError('Google Sign-In integration requires Google Identity Services SDK setup.');
    };

    return (
        <div className="relative min-h-screen w-full overflow-hidden bg-gradient-to-br from-slate-50 via-blue-50 to-slate-100 font-sans text-gray-800">
            {/* Background */}
            <div className="absolute inset-0 z-0">
                <img
                    src={bgImg}
                    alt="Background"
                    className="h-full w-full object-cover"
                />
                <div className="absolute inset-0 bg-black/10" />
            </div>

            {/* Content */}
            <div className="relative z-10 flex min-h-screen items-center justify-center px-4 py-10">
                <div
                    className={`relative w-full max-w-[400px] md:max-w-[450px] overflow-hidden rounded-3xl border bg-white/30 p-10 shadow-[0_20px_60px_rgba(0,0,0,0.25)] backdrop-blur-2xl transition-all duration-500 ${isHovering
                        ? 'border-blue-400/70 scale-[1.02] shadow-[0_30px_80px_rgba(0,51,102,0.35)]'
                        : 'border-white/50'
                        }`}
                    onMouseEnter={() => setIsHovering(true)}
                    onMouseLeave={() => setIsHovering(false)}
                >
                    {/* Glow Border */}
                    <div className="pointer-events-none absolute inset-0 -z-10 rounded-3xl bg-gradient-to-r from-blue-500 via-[#003366] to-blue-500 opacity-20 blur-2xl" />

                    {/* Header */}
                    <div className="mb-8 flex flex-col items-center text-center">
                        <div className="group relative mb-6">
                            <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-blue-400 to-[#003366] opacity-30 blur-xl transition-all duration-300 group-hover:opacity-60 group-hover:blur-2xl" />

                            <div className="relative rounded-2xl bg-white/20 p-4 backdrop-blur-md transition-all duration-300">
                                <img
                                    src={logoImg}
                                    alt="Logo"
                                    className="h-auto w-48 object-contain transition-transform duration-300"
                                />
                            </div>
                        </div>

                        <p className="text-slate-600 text-lg">
                            Create your account
                        </p>
                    </div>

                    {/* Error Message */}
                    {signupError && (
                        <div className="mb-4 p-3 rounded-xl bg-red-50 border border-red-200 text-red-600 text-sm font-medium text-center animate-pulse">
                            {signupError}
                        </div>
                    )}

                    {/* Form */}
                    <form className="space-y-5" onSubmit={handleSubmit}>
                        {/* Full Name */}
                        <div>
                            <label
                                htmlFor="fullName"
                                className={`mb-1 block text-sm font-semibold text-left transition-all duration-300 ${focusedField === 'fullName'
                                    ? 'text-[#003366]'
                                    : 'text-[#003366]'
                                    }`}
                            >
                                Full Name
                            </label>

                            <div className="relative">
                                <input
                                    id="fullName"
                                    type="text"
                                    required
                                    placeholder="Enter your full name"
                                    value={formData.fullName}
                                    onChange={handleChange}
                                    onFocus={() => setFocusedField('fullName')}
                                    onBlur={() => setFocusedField(null)}
                                    className={`w-full rounded-xl py-3 px-5 shadow-lg ring-2 transition-all duration-300 placeholder:text-gray-400 ${focusedField === 'fullName'
                                        ? 'bg-white ring-blue-500 shadow-[0_0_20px_rgba(59,130,246,0.35)] scale-[1.02]'
                                        : 'bg-white/70 ring-transparent hover:bg-white hover:shadow-xl'
                                        }`}
                                />
                            </div>
                        </div>

                        {/* Email */}
                        <div>
                            <label
                                htmlFor="email"
                                className={`mb-1 block text-sm font-semibold text-left transition-all duration-300 ${focusedField === 'email'
                                    ? 'text-[#003366]'
                                    : 'text-[#003366]'
                                    }`}
                            >
                                Email
                            </label>

                            <div className="relative">
                                <input
                                    id="email"
                                    type="email"
                                    required
                                    placeholder="Enter your email"
                                    value={formData.email}
                                    onChange={handleChange}
                                    onFocus={() => setFocusedField('email')}
                                    onBlur={() => setFocusedField(null)}
                                    className={`w-full rounded-xl py-3 px-5 shadow-lg ring-2 transition-all duration-300 placeholder:text-gray-400 ${focusedField === 'email'
                                        ? 'bg-white ring-blue-500 shadow-[0_0_20px_rgba(59,130,246,0.35)] scale-[1.02]'
                                        : 'bg-white/70 ring-transparent hover:bg-white hover:shadow-xl'
                                        }`}
                                />
                            </div>
                        </div>

                        {/* Password */}
                        <div>
                            <label
                                htmlFor="password"
                                className={`mb-1 block text-sm font-semibold text-left transition-all duration-300 ${focusedField === 'password'
                                    ? 'text-[#003366]'
                                    : 'text-[#003366]'
                                    }`}
                            >
                                Password
                            </label>

                            <div className="relative">
                                <input
                                    id="password"
                                    type="password"
                                    required
                                    placeholder="Enter password"
                                    value={formData.password}
                                    onChange={handleChange}
                                    onFocus={() => setFocusedField('password')}
                                    onBlur={() => setFocusedField(null)}
                                    className={`w-full rounded-xl py-3 px-5 shadow-lg ring-2 transition-all duration-300 placeholder:text-gray-400 ${focusedField === 'password'
                                        ? 'bg-white ring-blue-500 shadow-[0_0_20px_rgba(59,130,246,0.35)] scale-[1.02]'
                                        : 'bg-white/70 ring-transparent hover:bg-white hover:shadow-xl'
                                        }`}
                                />
                            </div>
                        </div>

                        {/* Button */}
                        <button
                            type="submit"
                            disabled={signupLoading}
                            className={`group relative flex w-full justify-center overflow-hidden rounded-xl px-4 py-3 text-base font-bold text-white shadow-[0_15px_50px_rgba(0,51,102,0.5)] transition-all duration-300 active:scale-95 ${signupLoading
                                ? 'bg-gray-400 cursor-not-allowed'
                                : 'bg-gradient-to-r from-[#003366] via-blue-700 to-[#003366] hover:shadow-[0_25px_70px_rgba(0,51,102,0.65)]'
                                }`}
                        >
                            <span className="relative z-10 flex items-center gap-2">
                                {signupLoading ? (
                                    <>
                                        <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                        </svg>
                                        Creating Account...
                                    </>
                                ) : (
                                    'Create Account'
                                )}
                            </span>

                            {!signupLoading && (
                                <div className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/30 to-transparent transition-transform duration-700 group-hover:translate-x-full" />
                            )}
                        </button>
                    </form>

                    {/* Divider */}
                    <div className="my-6 flex items-center gap-4 before:h-px before:flex-1 before:bg-slate-200 after:h-px after:flex-1 after:bg-slate-200">
                        <span className="text-sm font-medium text-slate-400">or</span>
                    </div>

                    {/* Google Button */}
                    <button
                        type="button"
                        onClick={handleGoogleAuth}
                        className="flex w-full items-center justify-center gap-3 rounded-xl border-2 border-slate-100 bg-white px-4 py-3 text-sm font-bold text-slate-700 shadow-sm transition-all duration-300 hover:border-blue-100 hover:bg-slate-50 hover:shadow-md active:scale-95"
                    >
                        <svg className="h-5 w-5" viewBox="0 0 24 24">
                            <path
                                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                                fill="#4285F4"
                            />
                            <path
                                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                                fill="#34A853"
                            />
                            <path
                                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                                fill="#FBBC05"
                            />
                            <path
                                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                                fill="#EA4335"
                            />
                        </svg>
                        Continue with Google
                    </button>

                    {/* Login Link */}
                    <p className="mt-8 text-center text-sm font-medium text-slate-800">
                        Already have an account?{' '}
                        <Link
                            to="/"
                            className="font-bold text-[#003366] underline decoration-2 underline-offset-4 transition hover:text-blue-700"
                        >
                            Login
                        </Link>
                    </p>
                </div>
            </div>

        </div>
    );
}
