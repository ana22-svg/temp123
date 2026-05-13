import React, { useState, useEffect, useCallback, useRef } from 'react';
import { ShieldCheck, TrendingUp, DollarSign, Percent, AlertCircle, Loader2, LayoutDashboard, BarChart3, UploadCloud, FileText, CheckCircle2, Zap, BrainCircuit, Activity, ChevronDown, ChevronUp } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

function cn(...inputs) { return twMerge(clsx(inputs)); }

const GRADE_MAPPING = { 'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6 };
const PLOTS = [
  "1_loan_amount_dist.png", "2_correlation_heatmap.png", "3_regressor_r2_comparison.png",
  "4_actual_vs_predicted.png", "5_regressor_cv_comparison.png", "6_feature_importance_regressor.png",
  "7_risk_category_distribution.png", "8_classifier_accuracy.png", "9_confusion_matrix.png",
  "10_feature_importance_classifier.png", "11_kmeans_elbow.png", "12_cluster_scatter.png",
  "13_f1_score_comparison.png", "14_grade_vs_risk.png"
];

const API_URL = 'http://localhost:5000';

function App() {
  const [activeTab, setActiveTab] = useState('predictor');

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500/30">
      {/* Background glow effects */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-[20%] -left-[10%] w-[50%] h-[50%] rounded-full bg-indigo-600/10 blur-[120px]" />
        <div className="absolute top-[60%] -right-[10%] w-[40%] h-[50%] rounded-full bg-purple-600/10 blur-[120px]" />
      </div>

      <nav className="relative z-20 border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0">
        <div className="container mx-auto px-4">
          <div className="flex flex-col sm:flex-row items-center justify-between h-auto sm:h-16 py-4 sm:py-0 gap-4">
            <div className="flex items-center gap-2 text-indigo-400 font-bold text-xl tracking-tight">
              <ShieldCheck className="w-6 h-6" />
              <span>RiskAI Pro</span>
            </div>
            <div className="flex gap-2 bg-slate-800/80 p-1 rounded-xl border border-slate-700/50">
              <TabButton active={activeTab === 'predictor'} onClick={() => setActiveTab('predictor')} icon={<Activity size={18}/>}>Predictor</TabButton>
              <TabButton active={activeTab === 'insights'} onClick={() => setActiveTab('insights')} icon={<BarChart3 size={18}/>}>Insights</TabButton>
            </div>
          </div>
        </div>
      </nav>
      
      <main className="flex-grow container mx-auto px-4 py-8 relative z-10 flex flex-col items-center">
        {activeTab === 'predictor' && <PredictorTab />}
        {activeTab === 'insights' && <InsightsTab />}
      </main>
    </div>
  );
}

function TabButton({ active, onClick, icon, children }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200",
        active ? "bg-slate-700 text-white shadow-sm" : "text-slate-400 hover:text-slate-200 hover:bg-slate-700/50"
      )}
    >
      {icon}
      {children}
    </button>
  );
}

function PredictorTab() {
  const [formData, setFormData] = useState({
    loan_amnt: 1500,
    installment: 50,
    dti: 15.5,
    grade: 'C'
  });
  const [isBehavioral, setIsBehavioral] = useState(false);
  const [behavioralData, setBehavioralData] = useState({
    spending_volatility: 'medium', // low, medium, high
    missed_payments: 0
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  // Debounce logic for real-time slider updates
  const timeoutRef = useRef(null);

  const fetchPrediction = useCallback(async (currentData, currentBehavioral, useBehavioral) => {
    setLoading(true);
    setError('');
    try {
      const payload = {
        loan_amnt: parseFloat(currentData.loan_amnt),
        installment: parseFloat(currentData.installment),
        dti: parseFloat(currentData.dti),
        grade_encoded: GRADE_MAPPING[currentData.grade]
      };

<<<<<<< HEAD
      const response = await fetch('https://temp123-1jgs.onrender.com/predict', {
=======
      const response = await fetch(`${API_URL}/predict`, {
>>>>>>> cf38c2afaa7e6e71c2b14677b5feafea36383941
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) throw new Error('Prediction failed');
      const data = await response.json();
      
      // MOCK BEHAVIORAL MODIFIER (Simulating Behavior-Enhanced Model)
      if (useBehavioral) {
        let modifier = 0;
        if (currentBehavioral.spending_volatility === 'high') modifier += 15;
        if (currentBehavioral.spending_volatility === 'low') modifier -= 5;
        modifier += (currentBehavioral.missed_payments * 10);
        
        data.risk_score = Math.min(Math.max(data.risk_score + modifier, 0), 100);
        
        // Recalculate category locally
        if (data.risk_score < 33) data.risk_category = 'Low';
        else if (data.risk_score < 66) data.risk_category = 'Medium';
        else data.risk_category = 'High';
      }

      setResult(data);
    } catch (err) {
      console.error(err);
      setError('Failed to fetch prediction.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => {
      fetchPrediction(formData, behavioralData, isBehavioral);
    }, 500); // 500ms debounce
    return () => clearTimeout(timeoutRef.current);
  }, [formData, behavioralData, isBehavioral, fetchPrediction]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(p => ({ ...p, [name]: value }));
  };

  const handleBehavioralChange = (e) => {
    const { name, value } = e.target;
    setBehavioralData(p => ({ ...p, [name]: value }));
  };

  return (
    <div className="w-full max-w-6xl flex flex-col gap-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      {/* Controls Panel */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-2xl flex flex-col gap-6">
        <div className="flex justify-between items-center pb-4 border-b border-slate-800">
          <h2 className="text-xl font-bold flex items-center gap-2">
            <TrendingUp className="text-indigo-400"/> Input Parameters
          </h2>
          
          {/* Model Toggle */}
          <div className="flex items-center gap-3 bg-slate-950 p-1.5 rounded-full border border-slate-800">
            <button 
              onClick={() => setIsBehavioral(false)}
              className={cn("px-4 py-1.5 rounded-full text-xs font-semibold transition-all", !isBehavioral ? "bg-indigo-500 text-white" : "text-slate-400 hover:text-white")}
            >
              Traditional
            </button>
            <button 
              onClick={() => setIsBehavioral(true)}
              className={cn("px-4 py-1.5 rounded-full text-xs font-semibold flex items-center gap-1 transition-all", isBehavioral ? "bg-purple-600 text-white shadow-[0_0_15px_rgba(147,51,234,0.5)]" : "text-slate-400 hover:text-white")}
            >
              <BrainCircuit size={14}/> Behavior-Enhanced
            </button>
          </div>
        </div>

        <div className="space-y-6">
          <SliderInput label="Loan Amount ($)" name="loan_amnt" min="100" max="5000" step="100" value={formData.loan_amnt} onChange={handleInputChange} description="Total principal amount requested by the borrower." />
          <SliderInput label="Monthly Installment ($)" name="installment" min="10" max="200" step="5" value={formData.installment} onChange={handleInputChange} description="Fixed monthly payment to repay the loan." />
          <SliderInput label="Debt-to-Income (%)" name="dti" min="0" max="40" step="0.5" value={formData.dti} onChange={handleInputChange} description="Ratio of total monthly debt payments to total monthly income." />
          
          <div className="space-y-2">
            <label className="text-sm font-semibold text-slate-300">Credit Grade</label>
            <select name="grade" value={formData.grade} onChange={handleInputChange} className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-200 outline-none focus:border-indigo-500 transition-colors">
              {Object.keys(GRADE_MAPPING).map(g => <option key={g} value={g}>Grade {g}</option>)}
            </select>
            <p className="text-xs text-slate-500">Assigned by the credit bureau (A is best, G is worst).</p>
          </div>
        </div>

        {isBehavioral && (
          <div className="mt-4 p-5 border border-purple-500/30 bg-purple-500/5 rounded-2xl space-y-5 animate-in slide-in-from-top-2">
            <h3 className="text-sm font-bold text-purple-400 uppercase tracking-wider flex items-center gap-2">
              <Zap size={16}/> Behavioral Indicators
            </h3>
            
            <div className="space-y-2">
              <label className="text-sm font-semibold text-slate-400">Spending Volatility</label>
              <div className="flex gap-2">
                {['low', 'medium', 'high'].map(v => (
                  <button key={v} name="spending_volatility" value={v} onClick={() => setBehavioralData(p => ({...p, spending_volatility: v}))}
                    className={cn("flex-1 py-2 rounded-lg text-sm font-medium border transition-colors capitalize", behavioralData.spending_volatility === v ? "bg-purple-600 border-purple-500 text-white" : "bg-slate-900 border-slate-700 text-slate-400 hover:bg-slate-800")}
                  >
                    {v}
                  </button>
                ))}
              </div>
            </div>

            <SliderInput label="Missed Payments (Last 6 Months)" name="missed_payments" min="0" max="6" step="1" value={behavioralData.missed_payments} onChange={handleBehavioralChange} color="purple" description="Number of recent late payments across all credit accounts." />
          </div>
        )}
      </div>

      {/* Results Panel */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-2xl flex flex-col relative overflow-hidden">
        {loading && (
          <div className="absolute inset-0 bg-slate-900/50 backdrop-blur-sm z-10 flex items-center justify-center">
            <Loader2 className="w-8 h-8 animate-spin text-indigo-500" />
          </div>
        )}

        <h2 className="text-xl font-bold flex items-center gap-2 mb-8 border-b border-slate-800 pb-4">
          <ShieldCheck className="text-indigo-400"/> Risk Assessment
        </h2>

        {result ? (
          <div className="flex flex-col items-center flex-grow">
            {/* Animated Gauge Chart */}
            <div className="w-64 h-64 mb-4 relative">
              <CircularProgressbar 
                value={result.risk_score} 
                text={`${result.risk_score.toFixed(1)}%`}
                circleRatio={0.75}
                styles={buildStyles({
                  rotation: 1 / 2 + 1 / 8,
                  strokeLinecap: 'round',
                  pathTransitionDuration: 1.5,
                  pathColor: getGaugeColor(result.risk_score),
                  textColor: '#f8fafc',
                  trailColor: '#1e293b',
                })}
              />
              <div className="absolute bottom-6 left-0 right-0 text-center">
                <span className={cn("px-4 py-1.5 rounded-full text-sm font-bold tracking-wider uppercase border", getCategoryStyles(result.risk_category))}>
                  {result.risk_category} RISK
                </span>
              </div>
            </div>

            {/* Feature Importance Breakdown */}
            <div className="w-full mt-auto space-y-4">
              <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-2">Risk Factors Breakdown</h3>
              <p className="text-xs text-slate-500 mb-4">This section explains which parameters contributed to the score.</p>
              
              <FactorBar label="Debt-to-Income (DTI)" impact={formData.dti > 20 ? 'high' : formData.dti < 10 ? 'low' : 'neutral'} value={`${formData.dti}%`} />
              <FactorBar label="Credit Grade" impact={['E','F','G'].includes(formData.grade) ? 'high' : ['A','B'].includes(formData.grade) ? 'low' : 'neutral'} value={`Grade ${formData.grade}`} />
              <FactorBar label="Loan Burden" impact={(formData.loan_amnt / formData.installment) > 35 ? 'high' : 'neutral'} value={`$${formData.loan_amnt} / $${formData.installment}`} />
              
              {isBehavioral && (
                <>
                  <FactorBar label="Spending Volatility" impact={behavioralData.spending_volatility === 'high' ? 'high' : behavioralData.spending_volatility === 'low' ? 'low' : 'neutral'} value={behavioralData.spending_volatility} />
                  <FactorBar label="Missed Payments" impact={behavioralData.missed_payments > 0 ? 'high' : 'neutral'} value={behavioralData.missed_payments} />
                </>
              )}
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center flex-grow text-slate-500 gap-4">
            <Activity className="w-16 h-16 opacity-20" />
            <p>Adjust parameters to see real-time risk assessment</p>
          </div>
        )}
      </div>
      </div>
      <EducationalGlossary />
    </div>
  );
}

// Helper Components
function SliderInput({ label, name, min, max, step, value, onChange, color = "indigo", description }) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <label className="text-sm font-semibold text-slate-300">{label}</label>
        <input type="number" name={name} value={value} onChange={onChange} className={cn("bg-slate-950 border border-slate-800 rounded-lg px-3 py-1 text-right w-24 text-slate-200 focus:border-indigo-500 outline-none")} />
      </div>
      <input type="range" name={name} min={min} max={max} step={step} value={value} onChange={onChange} className={cn("w-full accent-indigo-500 cursor-pointer", color === 'purple' && "accent-purple-500")} />
      {description && <p className="text-xs text-slate-500 mt-1">{description}</p>}
    </div>
  );
}

function FactorBar({ label, impact, value }) {
  const colorMap = {
    high: 'bg-rose-500',
    low: 'bg-emerald-500',
    neutral: 'bg-slate-600'
  };
  const iconMap = {
    high: 'text-rose-500',
    low: 'text-emerald-500',
    neutral: 'text-slate-500'
  };

  return (
    <div className="flex items-center justify-between bg-slate-950/50 p-3 rounded-xl border border-slate-800/50">
      <div className="flex items-center gap-3">
        <div className={cn("w-2 h-2 rounded-full", colorMap[impact])} />
        <span className="text-sm font-medium text-slate-300">{label}</span>
      </div>
      <span className={cn("text-sm font-bold capitalize", iconMap[impact])}>{value}</span>
    </div>
  );
}

function getGaugeColor(score) {
  if (score < 33) return '#10b981'; // emerald
  if (score < 66) return '#f59e0b'; // amber
  return '#f43f5e'; // rose
}

function getCategoryStyles(category) {
  const c = category?.toLowerCase() || '';
  if (c === 'low') return 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10';
  if (c === 'medium') return 'text-amber-400 border-amber-500/30 bg-amber-500/10';
  return 'text-rose-400 border-rose-500/30 bg-rose-500/10';
}


function InsightsTab() {
  return (
    <div className="w-full max-w-7xl animate-in fade-in slide-in-from-bottom-4">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold flex items-center gap-3"><BarChart3 className="text-indigo-400 w-8 h-8"/> Analytics Dashboard</h2>
          <p className="text-slate-400 mt-2">Visualizing the performance and features of the trained models.</p>
        </div>
      </div>



      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {PLOTS.map((plot, idx) => (
          <div key={idx} className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl hover:border-indigo-500/30 transition-colors group">
            <div className="h-64 overflow-hidden bg-slate-950 flex items-center justify-center p-2">
              <img 
                src={`${API_URL}/plots/${plot}`} 
                alt={plot} 
                className="max-h-full max-w-full object-contain group-hover:scale-105 transition-transform duration-500 rounded-xl"
                loading="lazy"
              />
            </div>
            <div className="p-4 border-t border-slate-800">
              <p className="text-sm font-semibold text-slate-300 truncate">
                {plot.replace(/_/g, ' ').replace('.png', '').replace(/^\d+\s*/, '')}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function EducationalGlossary() {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <div className="w-full bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl transition-all duration-300">
      <button onClick={() => setIsOpen(!isOpen)} className="w-full flex items-center justify-between text-left focus:outline-none group">
        <h3 className="text-xl font-bold flex items-center gap-2">
          <FileText className="text-indigo-400"/> Understanding the Metrics
        </h3>
        <div className="p-2 rounded-full bg-slate-800 group-hover:bg-slate-700 transition-colors">
          {isOpen ? <ChevronUp className="text-slate-300"/> : <ChevronDown className="text-slate-300"/>}
        </div>
      </button>
      
      {isOpen && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6 mt-6 pt-6 border-t border-slate-800 animate-in fade-in slide-in-from-top-4">
          <div className="bg-slate-950/80 p-5 rounded-2xl border border-slate-800/80 hover:border-indigo-500/30 transition-colors">
            <div className="w-8 h-8 rounded-full bg-indigo-500/10 flex items-center justify-center mb-3">
              <Percent className="w-4 h-4 text-indigo-400" />
            </div>
            <h4 className="font-bold text-slate-200 mb-2">Debt-to-Income (DTI)</h4>
            <p className="text-sm text-slate-400 leading-relaxed">The percentage of a borrower's gross monthly income that goes toward paying debts. A DTI over 20% is considered high risk.</p>
          </div>

          <div className="bg-slate-950/80 p-5 rounded-2xl border border-slate-800/80 hover:border-indigo-500/30 transition-colors">
            <div className="w-8 h-8 rounded-full bg-emerald-500/10 flex items-center justify-center mb-3">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
            </div>
            <h4 className="font-bold text-slate-200 mb-2">Credit Grade</h4>
            <p className="text-sm text-slate-400 leading-relaxed">A score assigned by credit bureaus evaluating credit history. 'A' is excellent, while 'G' indicates severe past payment issues.</p>
          </div>

          <div className="bg-slate-950/80 p-5 rounded-2xl border border-slate-800/80 hover:border-amber-500/30 transition-colors">
            <div className="w-8 h-8 rounded-full bg-amber-500/10 flex items-center justify-center mb-3">
              <TrendingUp className="w-4 h-4 text-amber-400" />
            </div>
            <h4 className="font-bold text-slate-200 mb-2">Loan Burden</h4>
            <p className="text-sm text-slate-400 leading-relaxed">The ratio of the total loan amount to the monthly installment. It roughly indicates how many months it will take to pay off the loan.</p>
          </div>

          <div className="bg-slate-950/80 p-5 rounded-2xl border border-slate-800/80 hover:border-purple-500/30 transition-colors">
            <div className="w-8 h-8 rounded-full bg-purple-500/10 flex items-center justify-center mb-3">
              <Zap className="w-4 h-4 text-purple-400" />
            </div>
            <h4 className="font-bold text-slate-200 mb-2">Spending Volatility</h4>
            <p className="text-sm text-slate-400 leading-relaxed">A behavioral metric tracking how erratically a person spends money month-to-month. High volatility suggests financial instability.</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
