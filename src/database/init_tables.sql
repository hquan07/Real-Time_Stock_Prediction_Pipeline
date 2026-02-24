--
-- PostgreSQL database dump
--

\restrict PJ1BAkSyIJl1UdqDofio57TZ9XZQsuc7XRYiBoC451pCKdxStyarfJBgIhof4id

-- Dumped from database version 15.14 (Debian 15.14-1.pgdg13+1)
-- Dumped by pg_dump version 15.14 (Debian 15.14-1.pgdg13+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: analyst_recommendations; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.analyst_recommendations (
    id bigint NOT NULL,
    ticker character varying(20),
    date date NOT NULL,
    firm text,
    recommendation text,
    target_price numeric
);


ALTER TABLE public.analyst_recommendations OWNER TO postgres;

--
-- Name: analyst_recommendations_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.analyst_recommendations_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.analyst_recommendations_id_seq OWNER TO postgres;

--
-- Name: analyst_recommendations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.analyst_recommendations_id_seq OWNED BY public.analyst_recommendations.id;


--
-- Name: balance_sheets; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.balance_sheets (
    id bigint NOT NULL,
    ticker character varying(20),
    period date NOT NULL,
    total_assets numeric,
    total_liabilities numeric,
    total_equity numeric,
    cash numeric,
    long_term_debt numeric
);


ALTER TABLE public.balance_sheets OWNER TO postgres;

--
-- Name: balance_sheets_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.balance_sheets_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.balance_sheets_id_seq OWNER TO postgres;

--
-- Name: balance_sheets_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.balance_sheets_id_seq OWNED BY public.balance_sheets.id;


--
-- Name: cashflow_statements; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cashflow_statements (
    id bigint NOT NULL,
    ticker character varying(20),
    period date NOT NULL,
    operating_cashflow numeric,
    investing_cashflow numeric,
    financing_cashflow numeric,
    free_cashflow numeric
);


ALTER TABLE public.cashflow_statements OWNER TO postgres;

--
-- Name: cashflow_statements_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cashflow_statements_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cashflow_statements_id_seq OWNER TO postgres;

--
-- Name: cashflow_statements_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.cashflow_statements_id_seq OWNED BY public.cashflow_statements.id;


--
-- Name: companies; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.companies (
    ticker character varying(20) NOT NULL,
    long_name text,
    sector text,
    industry text,
    country text,
    exchange text,
    website text,
    description text
);


ALTER TABLE public.companies OWNER TO postgres;

--
-- Name: dividends; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.dividends (
    id bigint NOT NULL,
    ticker character varying(20),
    date date NOT NULL,
    dividend numeric
);


ALTER TABLE public.dividends OWNER TO postgres;

--
-- Name: dividends_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.dividends_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.dividends_id_seq OWNER TO postgres;

--
-- Name: dividends_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.dividends_id_seq OWNED BY public.dividends.id;


--
-- Name: earnings_calendar; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.earnings_calendar (
    id bigint NOT NULL,
    ticker character varying(20),
    date date NOT NULL,
    eps_estimate numeric,
    revenue_estimate numeric
);


ALTER TABLE public.earnings_calendar OWNER TO postgres;

--
-- Name: earnings_calendar_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.earnings_calendar_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.earnings_calendar_id_seq OWNER TO postgres;

--
-- Name: earnings_calendar_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.earnings_calendar_id_seq OWNED BY public.earnings_calendar.id;


--
-- Name: earnings_history; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.earnings_history (
    id bigint NOT NULL,
    ticker character varying(20),
    period date NOT NULL,
    revenue_actual numeric,
    earnings_actual numeric,
    earnings_estimate numeric
);


ALTER TABLE public.earnings_history OWNER TO postgres;

--
-- Name: earnings_history_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.earnings_history_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.earnings_history_id_seq OWNER TO postgres;

--
-- Name: earnings_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.earnings_history_id_seq OWNED BY public.earnings_history.id;


--
-- Name: income_statements; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.income_statements (
    id bigint NOT NULL,
    ticker character varying(20),
    period date NOT NULL,
    total_revenue numeric,
    gross_profit numeric,
    operating_income numeric,
    net_income numeric,
    ebit numeric
);


ALTER TABLE public.income_statements OWNER TO postgres;

--
-- Name: income_statements_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.income_statements_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.income_statements_id_seq OWNER TO postgres;

--
-- Name: income_statements_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.income_statements_id_seq OWNED BY public.income_statements.id;


--
-- Name: institutional_holders; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.institutional_holders (
    id bigint NOT NULL,
    ticker character varying(20),
    holder text,
    shares bigint,
    date_reported date
);


ALTER TABLE public.institutional_holders OWNER TO postgres;

--
-- Name: institutional_holders_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.institutional_holders_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.institutional_holders_id_seq OWNER TO postgres;

--
-- Name: institutional_holders_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.institutional_holders_id_seq OWNED BY public.institutional_holders.id;


--
-- Name: key_metrics; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.key_metrics (
    id bigint NOT NULL,
    ticker character varying(20),
    date date NOT NULL,
    market_cap numeric,
    trailing_pe numeric,
    forward_pe numeric,
    beta numeric,
    roa numeric,
    roe numeric,
    profit_margin numeric,
    debt_to_equity numeric,
    current_ratio numeric
);


ALTER TABLE public.key_metrics OWNER TO postgres;

--
-- Name: key_metrics_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.key_metrics_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.key_metrics_id_seq OWNER TO postgres;

--
-- Name: key_metrics_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.key_metrics_id_seq OWNED BY public.key_metrics.id;


--
-- Name: major_holders; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.major_holders (
    id bigint NOT NULL,
    ticker character varying(20),
    holder_type text,
    percentage numeric
);


ALTER TABLE public.major_holders OWNER TO postgres;

--
-- Name: major_holders_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.major_holders_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.major_holders_id_seq OWNER TO postgres;

--
-- Name: major_holders_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.major_holders_id_seq OWNED BY public.major_holders.id;


--
-- Name: price_history; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.price_history (
    id bigint NOT NULL,
    ticker character varying(20),
    date date NOT NULL,
    open numeric,
    high numeric,
    low numeric,
    close numeric,
    adj_close numeric,
    volume bigint
);


ALTER TABLE public.price_history OWNER TO postgres;

--
-- Name: price_history_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.price_history_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.price_history_id_seq OWNER TO postgres;

--
-- Name: price_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.price_history_id_seq OWNED BY public.price_history.id;


--
-- Name: stock_splits; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.stock_splits (
    id bigint NOT NULL,
    ticker character varying(20),
    date date NOT NULL,
    split_ratio numeric
);


ALTER TABLE public.stock_splits OWNER TO postgres;

--
-- Name: stock_splits_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.stock_splits_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.stock_splits_id_seq OWNER TO postgres;

--
-- Name: stock_splits_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.stock_splits_id_seq OWNED BY public.stock_splits.id;


--
-- Name: analyst_recommendations id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.analyst_recommendations ALTER COLUMN id SET DEFAULT nextval('public.analyst_recommendations_id_seq'::regclass);


--
-- Name: balance_sheets id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.balance_sheets ALTER COLUMN id SET DEFAULT nextval('public.balance_sheets_id_seq'::regclass);


--
-- Name: cashflow_statements id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cashflow_statements ALTER COLUMN id SET DEFAULT nextval('public.cashflow_statements_id_seq'::regclass);


--
-- Name: dividends id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.dividends ALTER COLUMN id SET DEFAULT nextval('public.dividends_id_seq'::regclass);


--
-- Name: earnings_calendar id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.earnings_calendar ALTER COLUMN id SET DEFAULT nextval('public.earnings_calendar_id_seq'::regclass);


--
-- Name: earnings_history id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.earnings_history ALTER COLUMN id SET DEFAULT nextval('public.earnings_history_id_seq'::regclass);


--
-- Name: income_statements id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.income_statements ALTER COLUMN id SET DEFAULT nextval('public.income_statements_id_seq'::regclass);


--
-- Name: institutional_holders id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.institutional_holders ALTER COLUMN id SET DEFAULT nextval('public.institutional_holders_id_seq'::regclass);


--
-- Name: key_metrics id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.key_metrics ALTER COLUMN id SET DEFAULT nextval('public.key_metrics_id_seq'::regclass);


--
-- Name: major_holders id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.major_holders ALTER COLUMN id SET DEFAULT nextval('public.major_holders_id_seq'::regclass);


--
-- Name: price_history id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.price_history ALTER COLUMN id SET DEFAULT nextval('public.price_history_id_seq'::regclass);


--
-- Name: stock_splits id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stock_splits ALTER COLUMN id SET DEFAULT nextval('public.stock_splits_id_seq'::regclass);


--
-- Name: analyst_recommendations analyst_recommendations_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.analyst_recommendations
    ADD CONSTRAINT analyst_recommendations_pkey PRIMARY KEY (id);


--
-- Name: balance_sheets balance_sheets_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.balance_sheets
    ADD CONSTRAINT balance_sheets_pkey PRIMARY KEY (id);


--
-- Name: balance_sheets balance_sheets_ticker_period_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.balance_sheets
    ADD CONSTRAINT balance_sheets_ticker_period_key UNIQUE (ticker, period);


--
-- Name: cashflow_statements cashflow_statements_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cashflow_statements
    ADD CONSTRAINT cashflow_statements_pkey PRIMARY KEY (id);


--
-- Name: cashflow_statements cashflow_statements_ticker_period_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cashflow_statements
    ADD CONSTRAINT cashflow_statements_ticker_period_key UNIQUE (ticker, period);


--
-- Name: companies companies_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.companies
    ADD CONSTRAINT companies_pkey PRIMARY KEY (ticker);


--
-- Name: dividends dividends_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.dividends
    ADD CONSTRAINT dividends_pkey PRIMARY KEY (id);


--
-- Name: dividends dividends_ticker_date_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.dividends
    ADD CONSTRAINT dividends_ticker_date_key UNIQUE (ticker, date);


--
-- Name: earnings_calendar earnings_calendar_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.earnings_calendar
    ADD CONSTRAINT earnings_calendar_pkey PRIMARY KEY (id);


--
-- Name: earnings_calendar earnings_calendar_ticker_date_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.earnings_calendar
    ADD CONSTRAINT earnings_calendar_ticker_date_key UNIQUE (ticker, date);


--
-- Name: earnings_history earnings_history_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.earnings_history
    ADD CONSTRAINT earnings_history_pkey PRIMARY KEY (id);


--
-- Name: earnings_history earnings_history_ticker_period_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.earnings_history
    ADD CONSTRAINT earnings_history_ticker_period_key UNIQUE (ticker, period);


--
-- Name: income_statements income_statements_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.income_statements
    ADD CONSTRAINT income_statements_pkey PRIMARY KEY (id);


--
-- Name: income_statements income_statements_ticker_period_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.income_statements
    ADD CONSTRAINT income_statements_ticker_period_key UNIQUE (ticker, period);


--
-- Name: institutional_holders institutional_holders_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.institutional_holders
    ADD CONSTRAINT institutional_holders_pkey PRIMARY KEY (id);


--
-- Name: institutional_holders institutional_holders_ticker_holder_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.institutional_holders
    ADD CONSTRAINT institutional_holders_ticker_holder_key UNIQUE (ticker, holder);


--
-- Name: key_metrics key_metrics_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.key_metrics
    ADD CONSTRAINT key_metrics_pkey PRIMARY KEY (id);


--
-- Name: key_metrics key_metrics_ticker_date_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.key_metrics
    ADD CONSTRAINT key_metrics_ticker_date_key UNIQUE (ticker, date);


--
-- Name: major_holders major_holders_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.major_holders
    ADD CONSTRAINT major_holders_pkey PRIMARY KEY (id);


--
-- Name: price_history price_history_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.price_history
    ADD CONSTRAINT price_history_pkey PRIMARY KEY (id);


--
-- Name: price_history price_history_ticker_date_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.price_history
    ADD CONSTRAINT price_history_ticker_date_key UNIQUE (ticker, date);


--
-- Name: stock_splits stock_splits_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stock_splits
    ADD CONSTRAINT stock_splits_pkey PRIMARY KEY (id);


--
-- Name: stock_splits stock_splits_ticker_date_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stock_splits
    ADD CONSTRAINT stock_splits_ticker_date_key UNIQUE (ticker, date);


--
-- Name: idx_analyst_ticker_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_analyst_ticker_date ON public.analyst_recommendations USING btree (ticker, date);


--
-- Name: idx_price_history_ticker_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_price_history_ticker_date ON public.price_history USING btree (ticker, date);


--
-- Name: analyst_recommendations analyst_recommendations_ticker_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.analyst_recommendations
    ADD CONSTRAINT analyst_recommendations_ticker_fkey FOREIGN KEY (ticker) REFERENCES public.companies(ticker);


--
-- Name: balance_sheets balance_sheets_ticker_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.balance_sheets
    ADD CONSTRAINT balance_sheets_ticker_fkey FOREIGN KEY (ticker) REFERENCES public.companies(ticker);


--
-- Name: cashflow_statements cashflow_statements_ticker_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cashflow_statements
    ADD CONSTRAINT cashflow_statements_ticker_fkey FOREIGN KEY (ticker) REFERENCES public.companies(ticker);


--
-- Name: dividends dividends_ticker_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.dividends
    ADD CONSTRAINT dividends_ticker_fkey FOREIGN KEY (ticker) REFERENCES public.companies(ticker);


--
-- Name: earnings_calendar earnings_calendar_ticker_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.earnings_calendar
    ADD CONSTRAINT earnings_calendar_ticker_fkey FOREIGN KEY (ticker) REFERENCES public.companies(ticker);


--
-- Name: earnings_history earnings_history_ticker_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.earnings_history
    ADD CONSTRAINT earnings_history_ticker_fkey FOREIGN KEY (ticker) REFERENCES public.companies(ticker);


--
-- Name: income_statements income_statements_ticker_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.income_statements
    ADD CONSTRAINT income_statements_ticker_fkey FOREIGN KEY (ticker) REFERENCES public.companies(ticker);


--
-- Name: institutional_holders institutional_holders_ticker_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.institutional_holders
    ADD CONSTRAINT institutional_holders_ticker_fkey FOREIGN KEY (ticker) REFERENCES public.companies(ticker);


--
-- Name: key_metrics key_metrics_ticker_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.key_metrics
    ADD CONSTRAINT key_metrics_ticker_fkey FOREIGN KEY (ticker) REFERENCES public.companies(ticker);


--
-- Name: major_holders major_holders_ticker_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.major_holders
    ADD CONSTRAINT major_holders_ticker_fkey FOREIGN KEY (ticker) REFERENCES public.companies(ticker);


--
-- Name: price_history price_history_ticker_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.price_history
    ADD CONSTRAINT price_history_ticker_fkey FOREIGN KEY (ticker) REFERENCES public.companies(ticker);


--
-- Name: stock_splits stock_splits_ticker_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stock_splits
    ADD CONSTRAINT stock_splits_ticker_fkey FOREIGN KEY (ticker) REFERENCES public.companies(ticker);


--
-- PostgreSQL database dump complete
--

\unrestrict PJ1BAkSyIJl1UdqDofio57TZ9XZQsuc7XRYiBoC451pCKdxStyarfJBgIhof4id

