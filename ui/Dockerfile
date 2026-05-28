FROM nginx:alpine

# Copy custom Nginx configuration to disable browser caching
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy all static assets into Nginx html directory
COPY . /usr/share/nginx/html/

EXPOSE 80
