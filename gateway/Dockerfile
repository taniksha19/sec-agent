# Stage 1: compile the gateway into one static binary.
FROM golang:1.26.11-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY *.go ./
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /gateway .

# Stage 2: just the binary, on an image with no shell, running as a non-root user.
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /gateway /gateway
EXPOSE 8080
ENTRYPOINT ["/gateway"]
