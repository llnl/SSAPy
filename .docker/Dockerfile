FROM quay.io/centos/centos:stream9

ARG BUILD_DATE
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="SSAPy Docker image" \
      org.label-schema.description="Provides an environment containing a working copy of SSAPy." \
      org.label-schema.vendor="LLNL"

# Copy over SSAPy
COPY SSAPy/ /opt/SSAPy/

# Install C/C++ compilers
RUN echo "sslverify=False" >> /etc/dnf/dnf.conf && \
    dnf -y update && \
    dnf groupinstall --exclude=selinux-policy --exclude=selinux-policy-targeted -y "Development Tools" && \
    dnf -y clean all && \
    sed -i '$ d' /etc/dnf/dnf.conf && \
    gcc --version

# Install necessary cmake packages
RUN echo "sslverify=False" >> /etc/dnf/dnf.conf && \
    dnf install -y make cmake3 tree wget vi && \
    dnf -y clean all && \
    sed -i '$ d' /etc/dnf/dnf.conf && \
    make --version && \
    cmake --version && \
    ls -l /

# Install Python and some requirement dependencies
RUN echo "sslverify=False" >> /etc/dnf/dnf.conf && \
    dnf install -y epel-release && \
    dnf install -y python3.12 python3.12-devel python3.12-pip && \
    dnf -y clean all && \
    sed -i '$ d' /etc/dnf/dnf.conf && \
    unlink /usr/bin/python3 && \
    ln -s /usr/bin/python3.12 /usr/bin/python3 && \
    python3 --version

# Install the pip packages
RUN python3 -m pip --version && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --upgrade setuptools && \
    python3 -m pip install --no-cache-dir -r /opt/SSAPy/requirements.txt

# Install SSAPY
RUN ls -la / && \
    ls -la /opt && \
    ls -la /opt/SSAPy && \
    tree -r /opt/SSAPy && \
    cd /opt/SSAPy && \
    python3 setup.py build && \
    python3 setup.py install && \
    chmod -R go+rwX ./ && \
    chmod -R go+rwX /usr/lib64/python3.12/site-packages/ && \
    chmod -R go+rwX /usr/local/lib64/python3.12/site-packages/
