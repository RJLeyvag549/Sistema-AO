FROM node:18-slim

WORKDIR /app

# Instalamos dependencias básicas para un entorno de desarrollo React
COPY package*.json ./

RUN npm install

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--host"]
