const QuestionCard = ({ question, onNext, loading }) => {
    return (
        <div className="bg-white/80 backdrop-blur-sm p-10 rounded-2xl flex flex-col shadow-lg border-2 border-blue-100 min-h-[full] hover:shadow-xl transition-shadow duration-300">
            {/* Question */}
            <div className="flex items-start gap-4 w-full mb-8">
                <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-[#003380] to-[#0056b3] flex items-center justify-center shadow-md">
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                </div>
                <h2 className="text-2xl sm:text-3xl font-semibold text-gray-800 leading-relaxed">
                    {question?.question_text || question?.question || 'Loading question...'}
                </h2>
            </div>

            {/* Next Question Button */}
            <div className="flex justify-end mt-auto">
                <button
                    onClick={onNext}
                    disabled={loading}
                    className={`px-8 py-3 rounded-xl font-bold text-base transition-all duration-300 flex items-center gap-2
          ${loading
                            ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                            : 'bg-gradient-to-r from-[#003380] to-[#0056b3] hover:from-[#002260] hover:to-[#003d82] text-white shadow-lg hover:shadow-xl hover:scale-[1.02] active:scale-[0.98]'
                        }`}
                >
                    {loading ? (
                        <>
                            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                            </svg>
                            Loading...
                        </>
                    ) : (
                        <>
                            Next Question
                            <span>→</span>
                        </>
                    )}
                </button>
            </div>
        </div>
    );
};

export default QuestionCard;
