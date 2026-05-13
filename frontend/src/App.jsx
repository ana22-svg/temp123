import React, { useState } from 'react';
import { ShieldCheck, TrendingUp, DollarSign, Percent, AlertCircle, Loader2 } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/** Utility for Tailwind class merging */
function cn(...inputs) {
  return twMerge(clsx(inputs));
}

const GRADE_MAPPING = {
  'A': 0,
  'B': 1,
  'C': 2,
  'D': 3,
  'E': 4,
  'F': 5,
  'G': 6
};

function App() {
  const [formData, setFormData] = useState({
    loan_amnt: '',
    installment: '',
    dti: '',
    grade: 'A'
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    setError(''); // Clear error on typing
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setError('');
    setResult(null);

    // Basic Validation
    if (!formData.loan_amnt || !formData.installment || !formData.dti) {
      setError('Please fill in all numerical fields.');
      return;
    }
    
    if (parseFloat(formData.loan_amnt) <= 0 || parseFloat(formData.installment) <= 0 || parseFloat(formData.dti) < 0) {
      setError('Values must be positive numbers.');
      return;
    }

    setLoading(true);

    try {
      const payload = {
        loan_amnt: parseFloat(formData.loan_amnt),
        installment: parseFloat(formData.installment),
        dti: parseFloat(formData.dti),
        grade_encoded: GRADE_MAPPING[formData.grade]
      };

      const response = await fetch('https://temp123-1jgs.onrender.com/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error('Failed to get prediction from server.');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'An unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (category) => {
    if (!category) return 'text-gray-500';
    const lower = category.toLowerCase();
    if (lower.includes('low')) return 'text-emerald-500';
    if (lower.includes('medium')) return 'text-amber-500';
    if (lower.includes('high')) return 'text-rose-500';
    return 'text-primary-600';
  };

  const getRiskBgColor = (category) => {
    if (!category) return 'bg-gray-50';
    const lower = category.toLowerCase();
    if (lower.includes('low')) return 'bg-emerald-50 border-emerald-100';
    if (lower.includes('medium')) return 'bg-amber-50 border-amber-100';
    if (lower.includes('high')) return 'bg-rose-50 border-rose-100';
    return 'bg-primary-50 border-primary-100';
  };

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center p-4 sm:p-8 font-sans selection:bg-primary-200">
      <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-[20%] -left-[10%] w-[50%] h-[50%] rounded-full bg-primary-200/40 blur-[100px]" />
        <div className="absolute top-[60%] -right-[10%] w-[40%] h-[50%] rounded-full bg-indigo-200/40 blur-[100px]" />
      </div>

      <div className="w-full max-w-xl bg-white/80 backdrop-blur-xl rounded-3xl shadow-xl shadow-slate-200/50 border border-white p-6 sm:p-10 relative z-10 transition-all duration-300">
        
        {/* Header */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-500 to-indigo-600 text-white shadow-lg shadow-primary-500/30 mb-6 transform -rotate-3 transition-transform hover:rotate-0 duration-300">
            <ShieldCheck className="w-8 h-8" />
          </div>
          <h1 className="text-3xl font-bold text-slate-800 mb-2 tracking-tight">Loan Risk Prediction</h1>
          <p className="text-slate-500 font-medium">Predict borrower risk using AI models</p>
        </div>

        <form onSubmit={handlePredict} className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            
            {/* Loan Amount Input */}
            <div className="space-y-2">
              <label htmlFor="loan_amnt" className="block text-sm font-semibold text-slate-700">
                Loan Amount
              </label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400 group-focus-within:text-primary-500 transition-colors">
                  <DollarSign className="h-5 w-5" />
                </div>
                <input
                  type="number"
                  id="loan_amnt"
                  name="loan_amnt"
                  value={formData.loan_amnt}
                  onChange={handleChange}
                  placeholder="1500"
                  className="block w-full pl-10 pr-4 py-3 bg-slate-50/50 border border-slate-200 rounded-xl text-slate-800 placeholder-slate-400 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all outline-none"
                />
              </div>
            </div>

            {/* Installment Input */}
            <div className="space-y-2">
              <label htmlFor="installment" className="block text-sm font-semibold text-slate-700">
                Monthly Installment
              </label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400 group-focus-within:text-primary-500 transition-colors">
                  <DollarSign className="h-5 w-5" />
                </div>
                <input
                  type="number"
                  id="installment"
                  name="installment"
                  value={formData.installment}
                  onChange={handleChange}
                  placeholder="45.50"
                  step="0.01"
                  className="block w-full pl-10 pr-4 py-3 bg-slate-50/50 border border-slate-200 rounded-xl text-slate-800 placeholder-slate-400 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all outline-none"
                />
              </div>
            </div>

            {/* DTI Input */}
            <div className="space-y-2">
              <label htmlFor="dti" className="block text-sm font-semibold text-slate-700">
                Debt-to-Income (DTI)
              </label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400 group-focus-within:text-primary-500 transition-colors">
                  <Percent className="h-5 w-5" />
                </div>
                <input
                  type="number"
                  id="dti"
                  name="dti"
                  value={formData.dti}
                  onChange={handleChange}
                  placeholder="15.5"
                  step="0.1"
                  className="block w-full pl-10 pr-4 py-3 bg-slate-50/50 border border-slate-200 rounded-xl text-slate-800 placeholder-slate-400 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all outline-none"
                />
              </div>
            </div>

            {/* Credit Grade Dropdown */}
            <div className="space-y-2">
              <label htmlFor="grade" className="block text-sm font-semibold text-slate-700">
                Credit Grade
              </label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400 group-focus-within:text-primary-500 transition-colors">
                  <TrendingUp className="h-5 w-5" />
                </div>
                <select
                  id="grade"
                  name="grade"
                  value={formData.grade}
                  onChange={handleChange}
                  className="block w-full pl-10 pr-10 py-3 bg-slate-50/50 border border-slate-200 rounded-xl text-slate-800 appearance-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all outline-none cursor-pointer"
                >
                  {Object.keys(GRADE_MAPPING).map((grade) => (
                    <option key={grade} value={grade}>
                      Grade {grade}
                    </option>
                  ))}
                </select>
                <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none text-slate-400">
                  <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 9l4-4 4 4m0 6l-4 4-4-4" />
                  </svg>
                </div>
              </div>
            </div>

          </div>

          {error && (
            <div className="flex items-center gap-2 p-4 text-sm text-rose-600 bg-rose-50 border border-rose-100 rounded-xl animate-in fade-in slide-in-from-top-2 duration-300">
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              <p>{error}</p>
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-4 px-6 rounded-xl text-white font-semibold text-lg bg-gradient-to-r from-primary-500 to-indigo-600 hover:from-primary-600 hover:to-indigo-700 focus:ring-4 focus:ring-primary-500/30 transition-all duration-300 shadow-lg shadow-primary-500/25 disabled:opacity-70 disabled:cursor-not-allowed flex justify-center items-center gap-2 group transform active:scale-[0.98]"
          >
            {loading ? (
              <>
                <Loader2 className="w-6 h-6 animate-spin" />
                Analyzing Data...
              </>
            ) : (
              <>
                Predict Risk
                <TrendingUp className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </>
            )}
          </button>
        </form>

        {/* Results Section */}
        {result && (
          <div className={cn(
            "mt-8 p-6 rounded-2xl border transition-all duration-500 animate-in fade-in zoom-in-95",
            getRiskBgColor(result.risk_category)
          )}>
            <div className="flex flex-col items-center text-center space-y-4">
              <div className="space-y-1">
                <p className="text-sm font-semibold uppercase tracking-wider text-slate-500">Risk Assessment</p>
                <h3 className={cn("text-4xl font-bold tracking-tight", getRiskColor(result.risk_category))}>
                  {result.risk_category || "Unknown"}
                </h3>
              </div>
              
              <div className="w-full h-px bg-slate-200/60 my-2" />
              
              <div className="flex flex-col items-center justify-center">
                <p className="text-sm text-slate-500 font-medium mb-1">Confidence / Risk Score</p>
                <div className="flex items-baseline gap-1">
                  <span className="text-5xl font-black text-slate-800 tracking-tighter">
                    {result.risk_score !== undefined ? Number(result.risk_score).toFixed(2) : '--'}
                  </span>
                  {result.risk_score !== undefined && <span className="text-slate-500 font-semibold">%</span>}
                </div>
              </div>
              
              {result.risk_score !== undefined && (
                <div className="w-full mt-4 bg-slate-200/60 rounded-full h-2.5 overflow-hidden">
                  <div 
                    className={cn(
                      "h-2.5 rounded-full transition-all duration-1000 ease-out",
                      getRiskBgColor(result.risk_category).replace('bg-', 'bg-').split(' ')[0] === 'bg-emerald-50' ? 'bg-emerald-500' :
                      getRiskBgColor(result.risk_category).replace('bg-', 'bg-').split(' ')[0] === 'bg-amber-50' ? 'bg-amber-500' :
                      getRiskBgColor(result.risk_category).replace('bg-', 'bg-').split(' ')[0] === 'bg-rose-50' ? 'bg-rose-500' : 'bg-primary-500'
                    )}
                    style={{ width: `${Math.min(Math.max(result.risk_score, 0), 100)}%` }}
                  ></div>
                </div>
              )}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}

export default App;
