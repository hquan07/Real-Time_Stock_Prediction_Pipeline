import { create } from 'zustand';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

const useStore = create((set, get) => ({
    // Tickers State
    tickers: [{ label: 'Apple Inc. (AAPL)', value: 'AAPL' }],
    selectedTicker: 'AAPL',
    isLoadingTickers: false,
    
    // Historical Data State
    historicalData: [],
    isLoadingHistorical: false,
    historicalPeriod: '3M',
    
    // ML Data State
    predictionsData: { historical: [], future: [] },
    featureImportance: [],
    isLoadingML: false,
    
    // Actions
    setSelectedTicker: (ticker) => set({ selectedTicker: ticker }),
    setHistoricalPeriod: (period) => set({ historicalPeriod: period }),
    
    fetchTickers: async () => {
        set({ isLoadingTickers: true });
        try {
            const response = await axios.get(`${API_BASE_URL}/tickers`);
            if (response.data && response.data.length > 0) {
                set({ tickers: response.data, isLoadingTickers: false });
            } else {
                set({ isLoadingTickers: false });
            }
        } catch (error) {
            console.error('Failed to fetch tickers', error);
            set({ isLoadingTickers: false });
        }
    },
    
    fetchHistoricalData: async (ticker, period) => {
        set({ isLoadingHistorical: true });
        try {
            const response = await axios.get(`${API_BASE_URL}/stock/${ticker}?period=${period}`);
            set({ historicalData: response.data, isLoadingHistorical: false });
        } catch (error) {
            console.error('Failed to fetch historical data', error);
            set({ historicalData: [], isLoadingHistorical: false });
        }
    },
    
    fetchMLData: async (ticker) => {
        set({ isLoadingML: true });
        try {
            const [predRes, featRes] = await Promise.all([
                axios.get(`${API_BASE_URL}/predictions/${ticker}?model=RandomForest`),
                axios.get(`${API_BASE_URL}/feature-importance/${ticker}`)
            ]);
            set({ 
                predictionsData: predRes.data || { historical: [], future: [] },
                featureImportance: featRes.data || [],
                isLoadingML: false 
            });
        } catch (error) {
            console.error('Failed to fetch ML data', error);
            set({ 
                predictionsData: { historical: [], future: [] },
                featureImportance: [],
                isLoadingML: false 
            });
        }
    }
}));

export default useStore;
