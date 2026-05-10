/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'ao-bg': '#0a0b1e',
        'ao-panel': '#151932',
        'ao-accent': '#00f2ff',
        'ao-plasma': '#7000ff',
      }
    },
  },
  plugins: [],
}
