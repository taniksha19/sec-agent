#!/bin/sh
# Format check, vet and tests. Run by `make test-gateway` inside the golang image.
set -e
unformatted=$(gofmt -l .)
if [ -n "$unformatted" ]; then
    echo "gofmt needed: $unformatted"
    exit 1
fi
go vet ./...
go test ./...
