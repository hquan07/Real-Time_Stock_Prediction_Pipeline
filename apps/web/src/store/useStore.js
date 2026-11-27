import { create } from 'zustand';
import { apiClient } from '../config/api';

const messageFrom = (error, fallback) => error?.response?.data?.detail || fallback;

const useStore = create((set) => ({
    tickers: [],
    selectedTicker: '',
    isLoadingTickers: false,
    tickersError: null,

    historicalData: [],
    historicalMeta: null,
    historicalError: null,
    isLoadingHistorical: false,
    historicalPeriod: '3M',

    predictionsData: { historical: [], future: [] },
    featureImportance: [],
    mlMeta: null,
    mlError: null,
    isLoadingML: false,

    setSelectedTicker: (ticker) => set({ selectedTicker: ticker }),
    setHistoricalPeriod: (period) => set({ historicalPeriod: period }),

    fetchTickers: async () => {
        set({ isLoadingTickers: true, tickersError: null });
        try {
            const response = await apiClient.get('/tickers');
            const tickers = response.data?.data || [];
            set((state) => ({
                tickers,
                selectedTicker: state.selectedTicker || tickers[0]?.value || '',
                isLoadingTickers: false,
            }));
        } catch (error) {
            set({
                tickers: [],
                selectedTicker: '',
                tickersError: messageFrom(error, 'Ticker catalog unavailable.'),
                isLoadingTickers: false,
            });
        }
    },

    fetchHistoricalData: async (ticker, period) => {
        if (!ticker) return;
        set({ isLoadingHistorical: true, historicalError: null });
        try {
            const response = await apiClient.get(`/stock/${ticker}`, { params: { period } });
            set({
                historicalData: response.data?.data || [],
                historicalMeta: response.data?.meta || null,
                isLoadingHistorical: false,
            });
        } catch (error) {
            set({
                historicalData: [],
                historicalMeta: null,
                historicalError: messageFrom(error, 'Market data unavailable.'),
                isLoadingHistorical: false,
            });
        }
    },

    fetchMLData: async (ticker) => {
        if (!ticker) return;
        set({ isLoadingML: true, mlError: null });
        try {
            const [predictionResponse, importanceResponse] = await Promise.all([
                apiClient.get(`/predictions/${ticker}`, { params: { model: 'random_forest' } }),
                apiClient.get(`/feature-importance/${ticker}`),
            ]);
            set({
                predictionsData: predictionResponse.data?.data || { historical: [], future: [] },
                featureImportance: importanceResponse.data?.data || [],
                mlMeta: predictionResponse.data?.meta || null,
                isLoadingML: false,
            });
        } catch (error) {
            set({
                predictionsData: { historical: [], future: [] },
                featureImportance: [],
                mlMeta: null,
                mlError: messageFrom(error, 'Validated model output unavailable.'),
                isLoadingML: false,
            });
        }
    },
}));

export default useStore;
