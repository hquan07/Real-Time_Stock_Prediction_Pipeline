-- Insert Seed Tickers into companies
INSERT INTO public.companies (ticker, long_name, sector, industry, country, exchange, is_active)
VALUES 
    ('AAPL', 'Apple Inc.', 'Technology', 'Consumer Electronics', 'US', 'NASDAQ', true),
    ('MSFT', 'Microsoft Corporation', 'Technology', 'Software—Infrastructure', 'US', 'NASDAQ', true),
    ('NVDA', 'NVIDIA Corporation', 'Technology', 'Semiconductors', 'US', 'NASDAQ', true),
    ('GOOGL', 'Alphabet Inc.', 'Communication Services', 'Internet Content & Information', 'US', 'NASDAQ', true),
    ('AMZN', 'Amazon.com, Inc.', 'Consumer Cyclical', 'Internet Retail', 'US', 'NASDAQ', true),
    ('META', 'Meta Platforms, Inc.', 'Communication Services', 'Internet Content & Information', 'US', 'NASDAQ', true),
    ('TSLA', 'Tesla, Inc.', 'Consumer Cyclical', 'Auto Manufacturers', 'US', 'NASDAQ', true),
    ('JPM', 'JPMorgan Chase & Co.', 'Financial Services', 'Banks—Diversified', 'US', 'NYSE', true),
    ('V', 'Visa Inc.', 'Financial Services', 'Credit Services', 'US', 'NYSE', true),
    ('WMT', 'Walmart Inc.', 'Consumer Defensive', 'Discount Stores', 'US', 'NYSE', true),
    ('SPY', 'SPDR S&P 500 ETF Trust', 'ETF', 'ETF', 'US', 'NYSEARCA', true),
    ('QQQ', 'Invesco QQQ Trust', 'ETF', 'ETF', 'US', 'NASDAQ', true)
ON CONFLICT (ticker) DO UPDATE SET is_active = true;

-- Insert default watchlist for 'default_user'
INSERT INTO public.user_watchlist (user_id, ticker)
SELECT 'default_user', ticker FROM public.companies
ON CONFLICT (user_id, ticker) DO NOTHING;
