#!/usr/bin/env bash

mkdir -p ~/.cloudvolume
mkdir -p ~/.cloudvolume/secrets

cat > ~/.cloudvolume/secrets/aws-secret.json << EOL
[default]
aws_access_key_id = ${AWS_ACCESS_KEY_ID}
aws_secret_access_key = ${AWS_SECRET_ACCESS_KEY}
EOL
