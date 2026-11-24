/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { 950: "#070C18", 900: "#0B1220", 850: "#0F1729", 800: "#131D33", 700: "#1C2842", 600: "#2A3858" },
        electric: { 400: "#5B9BFF", 500: "#3B82F6", 600: "#2563EB" },
        signal: { cyan: "#22D3EE", violet: "#8B5CF6", amber: "#F5A524", red: "#F2545B", green: "#2DD4A4" },
      },
      fontFamily: { sans: ["Inter", "system-ui", "sans-serif"], mono: ["JetBrains Mono", "ui-monospace", "monospace"] },
    },
  },
  plugins: [],
};
