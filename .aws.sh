#!/usr/bin/env bash

mkdir -p ~/.cloudvolume
mkdir -p ~/.cloudvolume/secrets

cat > ~/.cloudvolume/secrets/aws-secret.json << EOL
{
	"AWS_ACCESS_KEY_ID": ${AWS_ACCESS_KEY_ID},
	"AWS_SECRET_ACCESS_KEY": ${AWS_SECRET_ACCESS_KEY}
}
EOL
