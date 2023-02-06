mkdir -p ~/.cloudvolume
mkdir -p ~/.cloudvolume/secrets

cat > ~/.cloudvolume/secrets/aws-secret.json << EOL
{
    "AWS_ACCESS_KEY_ID": "$1",
    "AWS_SECRET_ACCESS_KEY": "$2"
}
EOL

cat ~/.cloudvolume/secrets/aws-secret.json | cat