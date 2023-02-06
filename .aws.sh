mkdir -p ~/.cloudvolume
mkdir -p ~/.cloudvolume/secrets

cat > ~/.cloudvolume/secrets/aws-secret.json << EOL
{
    "AWS_ACCESS_KEY_ID": "${{ secrets.AWS_KEY }}",
    "AWS_SECRET_ACCESS_KEY": "${{ secrets.AWS_SEC_KEY }}"
}
EOL

cat ~/.cloudvolume/secrets/aws-secret.json | cat