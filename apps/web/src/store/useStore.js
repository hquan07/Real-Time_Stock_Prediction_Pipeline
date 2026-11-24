import { create } from 'zustand';
import { apiClient } from '../config/api';

const useStore = create((set) => ({
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
            const response = await apiClient.get('/tickers');
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
            const response = await apiClient.get(`/stock/${ticker}`, { params: { period } });
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
                apiClient.get(`/predictions/${ticker}`, { params: { model: 'RandomForest' } }),
                apiClient.get(`/feature-importance/${ticker}`)
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
