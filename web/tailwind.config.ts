import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: {
          50: "#f6f7f9", 100: "#eceef2", 200: "#d5d9e2", 300: "#b0b8c9",
          400: "#8492aa", 500: "#65748f", 600: "#505d76", 700: "#424c60",
          800: "#394151", 900: "#333946", 950: "#1c2029",
        },
        accent: {
          50: "#eef4ff", 100: "#d9e4ff", 200: "#bccfff", 300: "#8eb0ff",
          400: "#5985fd", 500: "#345ef8", 600: "#1f3ded", 700: "#182dda",
          800: "#1a27b1", 900: "#1c288c", 950: "#161b55",
        },
      },
      fontFamily: {
        sans: ['ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI',
               'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: { "fade-up": "fade-up 0.35s ease-out both" },
    },
  },
  plugins: [],
};

export default config;
