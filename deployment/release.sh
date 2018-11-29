echo "Current version is: "
git describe --tags --always --first-parent --abbrev=0

#  version="USER INPUT"
read -p "Enter version number to release: " version

git hf release start $version
git hf release finish $version
