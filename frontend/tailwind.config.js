/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      colors: {
        surface: {
          DEFAULT: '#0a0a0f',
          50: '#0d0d14',
          100: '#12121a',
          200: '#1a1a24',
          300: '#22222e',
          400: '#2a2a38',
        },
        accent: {
          DEFAULT: '#3b82f6',
          dim: '#2563eb',
          bright: '#60a5fa',
          glow: '#93c5fd',
        },
        cyber: {
          green: '#22c55e',
          amber: '#f59e0b',
          red: '#ef4444',
          purple: '#a855f7',
        },
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(59, 130, 246, 0.2)' },
          '100%': { boxShadow: '0 0 20px rgba(59, 130, 246, 0.4)' },
        },
      },
    },
  },
  plugins: [],
}
